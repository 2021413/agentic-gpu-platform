"""Lua scripts: every multi-step queue operation that must not interleave.

Redis runs a script to completion with nothing else in between, which is the
only reason two consumers polling the same queue can never be handed the same
job. Anything that reads a value and then writes based on it lives here rather
than in Python, where a round trip between the two would be a race.

Conventions used by all of them:

* keys that can be named up front arrive through ``KEYS``; keys that depend on
  data only the script can read (which job it popped, which run it belongs to)
  are built from a prefix passed in ``ARGV`` — see ``keys.Keyspace`` for the
  Redis Cluster caveat that comes with it;
* a hash field that is absent and one holding the empty string mean the same
  thing, matching ``codecs``;
* they are written to be replay-safe: every one of them checks the state it is
  about to change, so a redelivered command is a no-op rather than corruption.
"""

from __future__ import annotations

__all__ = [
    "ACKNOWLEDGE",
    "CANCEL_RUN_JOBS",
    "CLAIM",
    "ENQUEUE",
    "LOCK_EXTEND",
    "LOCK_RELEASE",
    "RECLAIM_EXPIRED",
    "RELEASE",
    "RENEW",
    "WORKER_UPDATE",
]

CLEAR_LEASE = (
    "'assigned_worker_id', '', 'lease_token', '', 'lease_holder', '', "
    "'lease_acquired_at', '', 'lease_expires_at', ''"
)

ENQUEUE = """
-- KEYS: job hash, ready set for the job type, run index
-- ARGV: job id, queue score, then the job record as field/value pairs
local status = redis.call('HGET', KEYS[1], 'status')
if status == 'LEASED' or status == 'RUNNING' then
  -- Republishing a job somebody is holding would hand the same work to a
  -- second consumer; a redelivered enqueue must lose to the live lease.
  return 0
end
if redis.call('HGET', KEYS[1], 'acked') == '1' then
  -- The acknowledgement tombstone: this job is finished and must not come back.
  return 0
end
redis.call('HSET', KEYS[1], unpack(ARGV, 3))
redis.call('ZADD', KEYS[2], tonumber(ARGV[2]), ARGV[1])
redis.call('SADD', KEYS[3], ARGV[1])
return 1
"""

CLAIM = """
-- KEYS: leases set, then one ready set per requested job type
-- ARGV: job key prefix, now ms, expiry ms, lease token, consumer id,
--       now ISO-8601, expiry ISO-8601, ready set prefix, delayed set prefix,
--       comma-separated job types
-- Returns the job record as it was *before* the lease, so the caller can
-- replay the transition through the domain entity instead of trusting Lua.
local leases = KEYS[1]

-- Anything whose delay has elapsed rejoins its ready set at the score it
-- already had, so waiting never costs a job its place behind newer work of the
-- same priority. Done here rather than in a sweeper so that "becomes
-- claimable" is atomic with "is claimed": there is no instant at which a due
-- job belongs to neither set.
for job_type in string.gmatch(ARGV[10], '[^,]+') do
  local delayed_key = ARGV[9] .. job_type
  local due = redis.call('ZRANGEBYSCORE', delayed_key, '-inf', ARGV[2], 'LIMIT', 0, 64)
  for _, due_id in ipairs(due) do
    local due_key = ARGV[1] .. due_id
    if redis.call('EXISTS', due_key) == 1
       and redis.call('HGET', due_key, 'status') == 'QUEUED' then
      redis.call('ZADD', ARGV[8] .. job_type,
                 tonumber(redis.call('HGET', due_key, 'score')), due_id)
    end
    redis.call('ZREM', delayed_key, due_id)
  end
end

for _ = 1, 64 do
  local best_key, best_id, best_score
  for i = 2, #KEYS do
    local head = redis.call('ZRANGE', KEYS[i], 0, 0, 'WITHSCORES')
    if head[1] ~= nil then
      local score = tonumber(head[2])
      -- Lowest score wins: the score encodes priority first, age second.
      if best_score == nil or score < best_score then
        best_score = score
        best_id = head[1]
        best_key = KEYS[i]
      end
    end
  end
  if best_id == nil then
    return nil
  end
  redis.call('ZREM', best_key, best_id)
  local job_key = ARGV[1] .. best_id
  local before = redis.call('HGETALL', job_key)
  if #before > 0 and redis.call('HGET', job_key, 'status') == 'QUEUED' then
    local attempt = tonumber(redis.call('HGET', job_key, 'attempt')) or 0
    local started = redis.call('HGET', job_key, 'started_at')
    if not started or started == '' then
      started = ARGV[6]
    end
    redis.call('HSET', job_key,
      'status', 'LEASED',
      'attempt', attempt + 1,
      'assigned_worker_id', ARGV[5],
      'started_at', started,
      'lease_token', ARGV[4],
      'lease_holder', ARGV[5],
      'lease_acquired_at', ARGV[6],
      'lease_expires_at', ARGV[7])
    redis.call('ZADD', leases, tonumber(ARGV[3]), best_id)
    return before
  end
  -- The entry pointed at a job that was cancelled, expired or is no longer
  -- claimable: it is now out of the ready set, so look at the next one.
end
return nil
"""

RENEW = """
-- KEYS: job hash, leases set
-- ARGV: job id, lease token, now ms, new expiry ms, new expiry ISO-8601
-- Returns nil when the lease was lost, which the caller turns into
-- JobLeaseExpiredError: the holder must stop working and discard its result.
if redis.call('EXISTS', KEYS[1]) == 0 then
  return nil
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
  return nil
end
local expiry = redis.call('ZSCORE', KEYS[2], ARGV[1])
if expiry == false or tonumber(expiry) <= tonumber(ARGV[3]) then
  return nil
end
redis.call('HSET', KEYS[1], 'lease_expires_at', ARGV[5])
redis.call('ZADD', KEYS[2], tonumber(ARGV[4]), ARGV[1])
return {redis.call('HGET', KEYS[1], 'lease_acquired_at'),
        redis.call('HGET', KEYS[1], 'lease_holder')}
"""

ACKNOWLEDGE = f"""
-- KEYS: job hash, leases set
-- ARGV: job id, lease token, run index prefix, run index suffix, tombstone ttl ms
-- A wrong token is silently ignored: it means a previous holder is reporting
-- late on work that has already been reassigned, and turning that into an
-- error would only make its retry loop replay forever.
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
  return 0
end
local run_id = redis.call('HGET', KEYS[1], 'run_id')
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[1], 'acked', '1', {CLEAR_LEASE})
-- Kept briefly rather than deleted: the tombstone is what makes a redelivered
-- enqueue of an already finished job a no-op.
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[5]))
if run_id then
  redis.call('SREM', ARGV[3] .. run_id .. ARGV[4], ARGV[1])
end
return 1
"""

RELEASE = f"""
-- KEYS: job hash, leases set
-- ARGV: job id, lease token, requeue flag, ready set prefix, ready-at ms,
--       delayed set prefix
-- The requeue flag: '0' do not requeue, '1' claimable now, '2' claimable at
-- ARGV[5].
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
if redis.call('HGET', KEYS[1], 'lease_token') ~= ARGV[2] then
  return 0
end
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('HSET', KEYS[1], {CLEAR_LEASE})
if ARGV[3] == '2' then
  -- Queued, but not yet offered to anyone. The job keeps its score, so when the
  -- delay elapses it rejoins the ready set where it belongs rather than at the
  -- back: the wait is a pause, not a demotion.
  redis.call('HSET', KEYS[1], 'status', 'QUEUED')
  redis.call('ZADD', ARGV[6] .. redis.call('HGET', KEYS[1], 'type'),
             tonumber(ARGV[5]), ARGV[1])
elseif ARGV[3] == '1' then
  -- The stored score sends the job back to the position it had, so giving work
  -- back never costs it its place behind newer jobs of the same priority.
  redis.call('HSET', KEYS[1], 'status', 'QUEUED')
  redis.call('ZADD', ARGV[4] .. redis.call('HGET', KEYS[1], 'type'),
             tonumber(redis.call('HGET', KEYS[1], 'score')), ARGV[1])
else
  redis.call('HSET', KEYS[1], 'status', 'PENDING')
end
return 1
"""

RECLAIM_EXPIRED = f"""
-- KEYS: leases set
-- ARGV: now ms, limit, job key prefix, ready set prefix, now ISO-8601, tombstone ttl ms
-- The detection half of "a worker that disappears releases its work": the
-- leases set is scored by expiry, so finding lapsed leases is a range query.
local expired = redis.call('ZRANGEBYSCORE', KEYS[1], '-inf', ARGV[1],
                           'LIMIT', 0, tonumber(ARGV[2]))
local reclaimed = {{}}
for _, job_id in ipairs(expired) do
  redis.call('ZREM', KEYS[1], job_id)
  local job_key = ARGV[3] .. job_id
  if redis.call('EXISTS', job_key) == 1 then
    local attempt = tonumber(redis.call('HGET', job_key, 'attempt')) or 0
    local budget = tonumber(redis.call('HGET', job_key, 'max_attempts')) or 1
    redis.call('HSET', job_key, {CLEAR_LEASE})
    if attempt < budget then
      redis.call('HSET', job_key, 'status', 'QUEUED')
      redis.call('ZADD', ARGV[4] .. redis.call('HGET', job_key, 'type'),
                 tonumber(redis.call('HGET', job_key, 'score')), job_id)
    else
      redis.call('HSET', job_key,
        'status', 'DEAD',
        'failure_kind', 'INFRASTRUCTURE',
        'failure_reason', 'lease expired and retry budget exhausted',
        'completed_at', ARGV[5])
      redis.call('PEXPIRE', job_key, tonumber(ARGV[6]))
    end
    reclaimed[#reclaimed + 1] = job_id
  end
end
return reclaimed
"""

CANCEL_RUN_JOBS = f"""
-- KEYS: run index, leases set
-- ARGV: job key prefix, ready set prefix, tombstone ttl ms
-- Queued jobs leave the ready set; in-flight ones keep their record but lose
-- their lease, so the worker holding them fails its next renewal and stops.
-- No completion timestamp is written here: the queue has no clock of its own,
-- and when the cancellation happened is PostgreSQL's record to keep.
local cancelled = {{}}
for _, job_id in ipairs(redis.call('SMEMBERS', KEYS[1])) do
  local job_key = ARGV[1] .. job_id
  if redis.call('EXISTS', job_key) == 0 then
    redis.call('SREM', KEYS[1], job_id)
  else
    local status = redis.call('HGET', job_key, 'status')
    if status ~= 'SUCCEEDED' and status ~= 'DEAD' and status ~= 'CANCELLED' then
      redis.call('ZREM', ARGV[2] .. redis.call('HGET', job_key, 'type'), job_id)
      redis.call('ZREM', KEYS[2], job_id)
      redis.call('HSET', job_key,
        'status', 'CANCELLED',
        'failure_kind', 'CANCELLED',
        {CLEAR_LEASE})
      redis.call('PEXPIRE', job_key, tonumber(ARGV[3]))
      cancelled[#cancelled + 1] = job_id
    end
  end
end
return cancelled
"""

WORKER_UPDATE = """
-- KEYS: worker hash
-- ARGV: ttl ms (<= 0 leaves it alone), then field/value pairs
-- Never recreates a key: a heartbeat or a status change must not resurrect a
-- worker that deregistered or aged out between the read and the write.
if redis.call('EXISTS', KEYS[1]) == 0 then
  return 0
end
redis.call('HSET', KEYS[1], unpack(ARGV, 2))
local ttl = tonumber(ARGV[1])
if ttl > 0 then
  redis.call('PEXPIRE', KEYS[1], ttl)
end
return 1
"""

LOCK_RELEASE = """
-- KEYS: lock key -- ARGV: owner token
-- Compare-and-delete: releasing a lock whose ttl already handed it to someone
-- else would let two holders run at once, which is the one thing a lock owes.
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

LOCK_EXTEND = """
-- KEYS: lock key -- ARGV: owner token, new ttl ms
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""
