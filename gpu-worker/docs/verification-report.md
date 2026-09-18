# Rapport de vérification par exécution réelle — image worker GPU RunPod

Vérification faite le **2026-09-18** sur une machine **sans GPU**, Linux
6.12.95+deb12-amd64, Docker avec driver `docker` (BuildKit), 605 GB libres sur `/`.

Toutes les sorties ci-dessous sont des sorties réelles, tronquées quand elles
sont longues. Aucune commande git n'a été exécutée. Aucun fichier source, test,
Dockerfile ou script du dépôt n'a été modifié.

---

## Résumé exécutif

| # | Point | Verdict |
|---|-------|---------|
| 1 | Le build aboutit | **ÉCHEC** — le `Dockerfile` committé ne construit pas, pour **deux raisons indépendantes** |
| 2 | Aucun poids de modèle dans l'image | **PROUVÉ** (sur image de substitution) |
| 3 | Variables de cache sous `/runpod-volume` | **PROUVÉ** (sur image de substitution) |
| 4 | Commandes console + six scripts | **PROUVÉ** |
| 5 | Codes de sortie 3 / 2 / 2 | **PROUVÉ** |
| 6 | Protocole d'arguments NUL avec retour à la ligne | **PROUVÉ** (dans le conteneur) |
| 7 | Cycle de vie complet jusqu'à l'échec GPU | **PROUVÉ** |
| 8 | `exec` → PID 1 et propagation SIGTERM | **PROUVÉ** |

**Deux défauts bloquants** (D1, D2) et **un défaut mineur** (D3) sont décrits en
fin de document, avec leur reproduction exacte.

---

## Avertissement méthodologique : l'image vérifiée n'est pas celle du dépôt

Le `Dockerfile` committé **ne produit aucune image**. Pour ne pas rendre les
points 2 à 8 invérifiables, une **copie de substitution** a été écrite hors du
dépôt, dans un répertoire temporaire, et ne diffère du `Dockerfile` committé que
par **deux lignes**, chacune neutralisant exactement un des deux défauts
bloquants :

```console
$ diff /home/hugo/Documents/programGen/gpu-worker/Dockerfile /tmp/.../scratchpad/Dockerfile.verify
125c125
<     rm -rf /root/.cache/pip; \
---
>     rm -rf /root/.cache/pip /runpod-volume; \
138c138
<     found="$(find / -xdev \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit)"; \
---
>     found="$(find / -xdev \( -name '*.safetensors' -o -name '*.gguf' \) ! -path '/usr/local/lib/python3.12/dist-packages/compressed_tensors/*' -print -quit)"; \
```

Les points 2 à 8 portent donc sur cette image de substitution, taguée
`agentic-gpu-worker:verify`. Tout ce qu'ils prouvent reste valable pour l'image
réelle **si et seulement si** les deux défauts sont corrigés sans autre
changement. Le contenu de la couche worker (2,2 MB) est rigoureusement identique
dans les deux cas : les deux lignes modifiées ne touchent que la logique de
garde et un `rm -rf` supplémentaire.

---

## Prérequis : l'image de base

L'image de base a dû être téléchargée intégralement ; le `docker pull` lancé
avant cette session n'était pas terminé et a mis environ **40 minutes** à
4,5–5 MB/s.

```console
$ docker images | grep vllm
vllm/vllm-openai:v0.28.0-cu129   249ed60fdd67       24.2GB             0B

$ tail -3 pull.log
Digest: sha256:ac259a0111c6cf462a72e449962b84f7a624b5cbec24bd7d9ec3b67d40ffd1bf
Status: Downloaded newer image for vllm/vllm-openai:v0.28.0-cu129
docker.io/vllm/vllm-openai:v0.28.0-cu129
```

`24.2 GB` est la taille **décompressée sur disque** (les ~9,7 GB annoncés sont la
taille compressée au registre).

---

## 1. Le build — **ÉCHEC**

### Commande exacte

```console
$ cd /home/hugo/Documents/programGen/gpu-worker
$ docker build --no-cache -t agentic-gpu-worker:verify .
```

### Sortie réelle (image de base déjà présente localement)

```
#12 [5/7] RUN set -eux;     found="$(find / -xdev \( -name '*.safetensors' -o -name '*.gguf' \) -print -quit)"; ...
#12 0.261 + find / -xdev ( -name *.safetensors -o -name *.gguf ) -print -quit
#12 0.712 + found=/usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
#12 0.712 + [ -n /usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors ]
#12 0.712 + echo model weights in a layer: /usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
#12 0.712 model weights in a layer: /usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
#12 0.712 + exit 1
#12 ERROR: process "/bin/sh -c set -eux; ..." did not complete successfully: exit code: 1
```

**Code de retour : 1. Temps jusqu'à l'échec : 5 s** (image de base déjà locale ;
compter ~40 min de plus sur une machine froide, pour le seul `pull`).

**Ligne du Dockerfile en cause : ligne 138.**

Les étapes `[2/7]` à `[4/7]` réussissent — la couche worker s'installe
correctement :

```
#11 3.740 Successfully installed agentic-gpu-worker-1.0.0
#11 3.944 worker 1.0.0
#11 3.975 /usr/local/bin/worker-preflight
#11 3.975 /usr/local/bin/worker-prepare-model
#11 3.975 /usr/local/bin/worker-serve-args
#11 3.975 /usr/local/bin/worker-ready
#11 3.976 /usr/local/bin/worker-health
#11 3.976 /usr/local/bin/worker-smoke-test
#11 DONE 4.0s
```

C'est donc **uniquement** l'étape de garde `[5/7]` qui échoue. Voir **D1** et
**D2**.

### Build de substitution (deux lignes corrigées)

```console
$ docker build --no-cache -f /tmp/.../Dockerfile.verify -t agentic-gpu-worker:verify .
BUILD_RC=0 DURATION=5s

$ docker images agentic-gpu-worker:verify
agentic-gpu-worker:verify   38ca731a5074       24.2GB             0B
```

- **Durée du build : 5 s** avec `--no-cache`, image de base déjà locale.
  Aucune dépendance n'est téléchargée : `httpx` et `huggingface_hub` sont déjà
  fournis par l'image de base (`dependencies missing from the base image: none`),
  et `pip install --no-deps` ne fait que construire la roue locale.
- **Taille finale : 24 225 280 774 octets (24,2 GB).**
- **Delta par rapport à l'image de base : 2 200 375 octets, soit 2,2 MB.**

---

## 2. Aucun poids de modèle dans l'image — **PROUVÉ**

Vérification **indépendante** du `RUN` interne : export du conteneur et analyse
de l'archive depuis l'hôte.

```console
$ cid=$(docker create agentic-gpu-worker:verify)
$ docker export "$cid" | tar -tvf - > export-listing.txt
$ docker rm -f "$cid"
$ wc -l export-listing.txt
141553 export-listing.txt
```

### Fichiers correspondant à un motif de poids, triés par taille

```console
$ awk '$1 ~ /^-/ {n=""; for(i=6;i<=NF;i++) n=n $i " ";
       if (n ~ /\.(safetensors|gguf|bin|pt|pth|ckpt|onnx|npz|msgpack|h5)[ ]*$/)
         printf "%12d  %s\n", $3, n}' export-listing.txt | sort -rn | head

     1436901  usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
      247165  usr/local/lib/python3.12/dist-packages/sentencepiece/package_data/nfkc_cf.bin
      247164  usr/local/lib/python3.12/dist-packages/sentencepiece/package_data/nmt_nfkc_cf.bin
      240008  usr/local/lib/python3.12/dist-packages/sentencepiece/package_data/nfkc.bin
      240007  usr/local/lib/python3.12/dist-packages/sentencepiece/package_data/nmt_nfkc.bin
         453  usr/local/lib/python3.12/dist-packages/numpy/lib/tests/data/py3-objarr.npz
         366  usr/local/lib/python3.12/dist-packages/numpy/lib/tests/data/py2-objarr.npz
         195  usr/local/lib/python3.12/dist-packages/_cuda_bindings_redirector.pth
         151  usr/local/lib/python3.12/dist-packages/distutils-precedence.pth
         116  usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl_packages.pth

# nombre de fichiers de ce type dépassant 100 MB :
0
```

Le plus gros « poids » est de **1,4 MB** : ce sont les matrices de Hadamard
livrées par la bibliothèque `compressed_tensors`, pas des poids de modèle.
Aucun `.safetensors` / `.gguf` / `.bin` volumineux n'existe dans l'image.

### Les 10 plus gros fichiers de l'image

```console
$ awk '$1 ~ /^-/ {n=""; for(i=6;i<=NF;i++) n=n $i " "; printf "%12d  %s\n", $3, n}' \
    export-listing.txt | sort -rn | head -10

  1097254628  usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublasLt_static.a
  1050480753  usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so
   749210000  usr/local/cuda-12.9/targets/x86_64-linux/lib/libcublasLt.so.12.9.2.10
   749205904  usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib/libcublasLt.so.12
   515924288  usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib/libcudnn_engines_precompiled.so.9
   508146448  usr/local/lib/python3.12/dist-packages/vllm/_C_stable_libtorch.abi3.so
   486556368  usr/local/lib/python3.12/dist-packages/nvidia/cusparse/lib/libcusparse.so.12
   460782192  usr/local/lib/python3.12/dist-packages/triton/_C/libtriton.so
   438418713  usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cpu.so
   422386624  usr/local/lib/python3.12/dist-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so
```

Les dix plus gros fichiers sont des bibliothèques CUDA, PyTorch et vLLM. Les
24,2 GB sont donc intégralement imputables à l'image de base ; la couche worker
n'ajoute que **2,2 MB**.

**Verdict : l'invariant « aucun poids dans l'image » tient.** Mais la garde
censée le faire respecter est cassée dans les deux sens (D1 : faux positif ;
D2 : le build viole lui-même l'autre moitié de la garde).

---

## 3. Variables de cache vers le volume — **PROUVÉ**

```console
$ docker inspect agentic-gpu-worker:verify --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|VLLM_CACHE_ROOT|TORCH_HOME|TMPDIR|XDG_CACHE_HOME|TRITON_CACHE_DIR|PERSISTENT_ROOT)='

PERSISTENT_ROOT=/runpod-volume
HF_HOME=/runpod-volume/huggingface
HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub
HF_HUB_CACHE=/runpod-volume/huggingface/hub
VLLM_CACHE_ROOT=/runpod-volume/vllm
TORCH_HOME=/runpod-volume/torch
TMPDIR=/runpod-volume/tmp
XDG_CACHE_HOME=/runpod-volume/xdg
TRITON_CACHE_DIR=/runpod-volume/vllm/triton
```

Les **huit** variables demandées sont présentes et toutes sous `/runpod-volume`.

Configuration annexe vérifiée au passage :

```console
$ docker inspect agentic-gpu-worker:verify \
    --format 'Entrypoint={{json .Config.Entrypoint}} Cmd={{json .Config.Cmd}} WorkingDir={{.Config.WorkingDir}} Healthcheck={{json .Config.Healthcheck}}'

Entrypoint=["/usr/local/bin/entrypoint.sh"] Cmd=null WorkingDir=/vllm-workspace
Healthcheck={"Test":["CMD","worker-health"],"Interval":30000000000,"Timeout":10000000000,"StartPeriod":2700000000000,"Retries":3}
```

`Cmd` est bien `null` (pas de seconde source de vérité pour l'argv) et le
`start-period` vaut bien 2700 s = 45 min.

### `/runpod-volume` n'existe pas dans l'image

```console
$ grep -c 'runpod-volume' export-listing.txt
0
```

Zéro entrée dans les 141 553 du système de fichiers exporté.

**Attention** : ce résultat est obtenu grâce à la ligne 125 corrigée dans la
copie de substitution. Avec le `Dockerfile` committé, ce serait faux — voir
**D2**, qui le démontre par exécution.

---

## 4. Commandes console et scripts — **PROUVÉ**

### `worker-preflight --skip-storage`

```console
$ docker run --rm --entrypoint worker-preflight agentic-gpu-worker:verify --skip-storage

── configuration ───────────────────────────────────────────────
  model                  Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  revision               unpinned (resolves to main)
  served as              Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
  persistent root        /runpod-volume
  listen                 0.0.0.0:8000
  max model len          16384
  gpu memory utilisation 0.9
  tensor parallel        1
  min free disk          60 GB
  api key                NOT SET (open port)
  hf token               not set

── software ────────────────────────────────────────────────────
  vllm               0.28.0+cu129
  torch              2.13.0+cu129
  transformers       5.15.1
  huggingface-hub    1.28.0
  httpx              0.28.1
  base image         vllm/vllm-openai:v0.28.0-cu129

── hardware ────────────────────────────────────────────────────
  GPUs: none visible (nvidia-smi is not on PATH)
EXIT=0
```

Les versions réellement présentes dans l'image de base sont donc :
vLLM **0.28.0+cu129**, torch **2.13.0+cu129**, transformers **5.15.1**,
huggingface-hub **1.28.0**, httpx **0.28.1**.

### `worker-serve-args --lines`

```console
$ docker run --rm --entrypoint worker-serve-args agentic-gpu-worker:verify --lines
vllm
serve
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
--host
0.0.0.0
--port
8000
--served-model-name
Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
--max-model-len
16384
--gpu-memory-utilization
0.9
--tensor-parallel-size
1
EXIT=0
```

### `worker-health`

```console
$ docker run --rm --entrypoint worker-health agentic-gpu-worker:verify
ConnectError: [Errno 111] Connection refused
EXIT=5
```

Comportement correct : rien n'écoute, donc « pas vivant », code 5 conforme au
contrat, message exploitable.

### Les six scripts sont exécutables

```console
$ docker run --rm --entrypoint bash agentic-gpu-worker:verify -c \
    'for s in entrypoint.sh preflight.sh prepare-model.sh health.sh ready.sh smoke-test.sh; do
       if [ -x "/usr/local/bin/$s" ]; then printf "%-18s %s\n" "$s" "$(stat -c %A /usr/local/bin/$s) EXECUTABLE";
       else printf "%-18s MISSING or NOT EXECUTABLE\n" "$s"; fi; done'

entrypoint.sh      -rwxr-xr-x EXECUTABLE
preflight.sh       -rwxr-xr-x EXECUTABLE
prepare-model.sh   -rwxr-xr-x EXECUTABLE
health.sh          -rwxr-xr-x EXECUTABLE
ready.sh           -rwxr-xr-x EXECUTABLE
smoke-test.sh      -rwxr-xr-x EXECUTABLE
```

---

## 5. Modes de défaillance et codes de sortie — **PROUVÉ**

### 5a — sans volume monté : attendu 3

```console
$ out=$(docker run --rm agentic-gpu-worker:verify 2>&1); echo "EXIT=$?"
EXIT=3
```

```
── persistent storage ──────────────────────────────────────────

persistent storage error: /runpod-volume does not exist, so no persistent volume is attached
  path: /runpod-volume
  fix:  attach a RunPod network volume with mount path /runpod-volume, or set PERSISTENT_ROOT to where it is mounted

2026-09-18T13:17:02Z  ERROR 'worker-preflight' exited with 3
2026-09-18T13:17:02Z  ERROR persistent storage error — the volume is missing, read-only or full.
2026-09-18T13:17:02Z  ERROR   Check that a RunPod network volume is attached at ${PERSISTENT_ROOT}
2026-09-18T13:17:02Z  ERROR   (currently '/runpod-volume') and that it has room
2026-09-18T13:17:02Z  ERROR   for the weights. Do NOT work around this by running without a volume:
2026-09-18T13:17:02Z  ERROR   the model would be re-downloaded on every Pod start.
```

Message exploitable : il nomme le chemin, la cause et l'action.

### 5b — `MAX_MODEL_LEN=10` : attendu 2

```console
$ out=$(docker run --rm -e MAX_MODEL_LEN=10 agentic-gpu-worker:verify 2>&1); echo "EXIT=$?"
EXIT=2
```

```
2026-09-18T13:17:29Z  STAGE ══ PREFLIGHT ══
configuration error: MAX_MODEL_LEN must be at least 1024, got 10
  variable: MAX_MODEL_LEN

2026-09-18T13:17:30Z  ERROR 'worker-preflight' exited with 2
2026-09-18T13:17:30Z  ERROR configuration error — an environment variable is wrong or missing.
2026-09-18T13:17:30Z  ERROR   Retrying will not help. Fix the Pod template / .env and redeploy.
```

Noter que l'erreur est détectée **avant** la vérification de stockage : sans
volume monté, c'est bien 2 et non 3 qui remonte, donc l'ordre de priorité des
diagnostics est correct.

### 5c — `MODEL_DOWNLOAD_MAX_ATTEMPTS=0` : attendu 2

```console
$ out=$(docker run --rm -e MODEL_DOWNLOAD_MAX_ATTEMPTS=0 agentic-gpu-worker:verify 2>&1); echo "EXIT=$?"
EXIT=2
```

```
2026-09-18T13:17:31Z  STAGE ══ PREFLIGHT ══
configuration error: MODEL_DOWNLOAD_MAX_ATTEMPTS must be at least 1, got 0
  variable: MODEL_DOWNLOAD_MAX_ATTEMPTS
  fix:      a zero attempt budget means the worker never tries to download

2026-09-18T13:17:31Z  ERROR 'worker-preflight' exited with 2
```

Le garde-fou décrit dans le commentaire de `config.py` (« the timing group was
unvalidated ») est effectivement actif et le `hint` est présent.

---

## 6. Protocole d'arguments avec un mot contenant un retour à la ligne — **PROUVÉ**

### La commande telle que spécifiée échoue — pour une raison sans rapport

```console
$ docker run --rm -e WORKER_ALLOW_EPHEMERAL_STORAGE=1 -e PERSISTENT_ROOT=/tmp/v \
    -e VLLM_EXTRA_ARGS='--chat-template "ligne1
  ligne2"' --entrypoint bash agentic-gpu-worker:verify -c \
    'f=$(mktemp); worker-serve-args > "$f"; mapfile -d "" -t a < "$f"; echo "mots=${#a[@]}"; printf "dernier=[%s]\n" "${a[-1]}"'

mktemp: failed to create file via template '/runpod-volume/tmp/tmp.XXXXXXXXXX': No such file or directory
bash: line 1: : No such file or directory
mots=0
dernier=[]
```

Ce n'est **pas** un défaut du protocole d'arguments : `--entrypoint bash`
court-circuite `entrypoint.sh`, donc `TMPDIR=/runpod-volume/tmp` reste la valeur
gravée dans l'image et pointe sur un répertoire inexistant. `mktemp` (coreutils)
échoue sèchement là où `tempfile` de Python retomberait sur `/tmp`. Voir **D3**.

### Variante corrigée (`TMPDIR=/tmp`), qui teste réellement le protocole

```console
$ docker run --rm -e WORKER_ALLOW_EPHEMERAL_STORAGE=1 -e PERSISTENT_ROOT=/tmp/v -e TMPDIR=/tmp \
    -e VLLM_EXTRA_ARGS='--chat-template "ligne1
  ligne2"' --entrypoint bash agentic-gpu-worker:verify -c \
    'f=$(mktemp); worker-serve-args > "$f"; mapfile -d "" -t a < "$f";
     echo "mots=${#a[@]}"; printf "dernier=[%s]\n" "${a[-1]}";
     python3 -c "
from worker.config import WorkerConfig
v=WorkerConfig.from_env().vllm_argv()
print(\"vllm_argv_len=\",len(v)); print(\"vllm_argv_last=\",repr(v[-1]))";
     tr -dc "\0" < "$f" | wc -c; tail -c 24 "$f" | od -c'

mots=17
dernier=[ligne1
ligne2]
--- attendu par vllm_argv() ---
vllm_argv_len= 17
vllm_argv_last= 'ligne1\nligne2'
--- NULs dans le fichier ---
17
--- fin du fichier en octal ---
0000000   -   t   e   m   p   l   a   t   e  \0   l   i   g   n   e   1
0000020  \n   l   i   g   n   e   2  \0
0000030
```

Quatre preuves concordantes, toutes **dans le conteneur** :

1. `mots=17` et `vllm_argv_len=17` — le compte lu par bash **correspond**
   exactement à celui produit par `vllm_argv()`.
2. `dernier=[ligne1\nligne2]` s'affiche sur deux lignes : le retour à la ligne
   est **intact dans un seul mot**.
3. Le fichier contient exactement **17 octets NUL** — un par mot, aucun mot
   perdu ni scindé.
4. Le vidage octal montre littéralement `... -template \0 ligne1 \n ligne2 \0` :
   le séparateur est bien NUL, et le `\n` est à l'intérieur du dernier mot.

Le correctif « séparateur NUL au lieu de retour à la ligne » est donc
effectivement en place et effectif dans l'image.

---

## 7. Le cycle de vie démarre réellement — **PROUVÉ**

`WORKER_SKIP_MODEL_PREPARATION` est bien supporté par `docker/entrypoint.sh`
(accepte `1|true|yes|on`).

```console
$ mkdir -p /tmp/.../vol7
$ docker run --rm -v /tmp/.../vol7:/runpod-volume \
    -e WORKER_SKIP_MODEL_PREPARATION=1 agentic-gpu-worker:verify
```

### Trace complète des étapes franchies

```
2026-09-18T13:18:07Z  STAGE ══ BOOT ══
  host             c33d7f3e6baa
  base image       vllm/vllm-openai:v0.28.0-cu129
  worker version   1.0.0
  persistent root  /runpod-volume
  pid              1 (this shell is replaced by vLLM at the end)

2026-09-18T13:18:07Z  STAGE ══ PREFLIGHT ══
  [configuration / software — identiques au point 4]
── hardware ────────────────────────────────────────────────────
  GPUs: none visible (nvidia-smi is not on PATH)
── persistent storage ──────────────────────────────────────────
  persistent root        /runpod-volume
  separate mount         yes
  writable               yes
  capacity               585.4 GB free of 936.4 GB (32% used) on /runpod-volume
  model cache in use     0.0 GB
  required free          60.0 GB
── model ───────────────────────────────────────────────────────
  not prepared on this volume yet
  ! no GPU is visible: vLLM will fail to start
preflight passed

2026-09-18T13:18:07Z  STAGE ══ VOLUME ══
  caches point at the volume: HF_HUB_CACHE=/runpod-volume/huggingface/hub

2026-09-18T13:18:07Z  STAGE ══ MODEL PREPARATION ══
  WARN  WORKER_SKIP_MODEL_PREPARATION is set: the model will NOT be downloaded
  WARN  or verified. [...]

2026-09-18T13:18:07Z  STAGE ══ MODEL VALIDATION ══
model is not prepared
  WARN  no verified snapshot on the volume (expected: preparation was skipped)

2026-09-18T13:18:07Z  STAGE ══ vLLM START ══
  vLLM command (secrets redacted):
      vllm / serve / Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 / --host / 0.0.0.0 /
      --port / 8000 / --served-model-name / Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 /
      --max-model-len / 16384 / --gpu-memory-utilization / 0.9 /
      --tensor-parallel-size / 1

2026-09-18T13:18:08Z  STAGE ══ EXEC ══
  handing over to vLLM (pid 1 becomes PID 1 of the container)
```

Puis vLLM lui-même démarre et échoue, faute de GPU — ce qui est le résultat
attendu :

```
INFO 09-18 13:18:24 [importing.py:53] Triton is installed but 0 active driver(s) found (expected 1). Disabling Triton to prevent runtime errors.
W0918 13:18:25.289000 1 torch/utils/cpp_extension.py:178] No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
Traceback (most recent call last):
  File "/usr/local/bin/vllm", line 10, in <module>
    sys.exit(main())
  ...
  File "/usr/local/lib/python3.12/dist-packages/vllm/config/device.py", line 56, in __post_init__
    raise RuntimeError(
RuntimeError: Failed to infer device type, please set the environment variable `VLLM_LOGGING_LEVEL=DEBUG` to turn on verbose logging to help debug the issue.
```

**Les sept étapes du cycle de vie se sont déroulées sans erreur.** Le seul échec
est celui de vLLM à l'absence de GPU, et il survient **après** le `exec` ; le
code de sortie du conteneur (1) est celui de vLLM, pas un code du contrat
worker, ce qui est le comportement correct après passation.

Le message final est compréhensible, mais « Failed to infer device type » vient
de vLLM et est plus sec que les diagnostics du worker ; c'est acceptable, le
préflight ayant déjà averti `! no GPU is visible: vLLM will fail to start`.

### Preuve annexe : le volume a bien été utilisé

```console
$ find /tmp/.../vol7 -maxdepth 2 | sort
vol7/huggingface
vol7/huggingface/hub
vol7/logs
vol7/models
vol7/state
vol7/tmp
vol7/tmp/torchinductor_root      <-- écrit par torch, via TMPDIR
vol7/torch
vol7/vllm
vol7/vllm/triton
vol7/xdg
```

L'arborescence complète a été créée sur le volume, et `torch` y a effectivement
écrit `tmp/torchinductor_root` : la redirection des caches n'est pas seulement
déclarée, elle est **observée en fonctionnement**.

---

## 8. SIGTERM et PID 1 — **PROUVÉ**

### 8a — le test proposé (`exec sleep 300`) est trompeur

```console
$ cid=$(docker run -d --entrypoint bash agentic-gpu-worker:verify -c 'exec sleep 300')
$ docker exec $cid ps -o pid,comm -p 1
    PID COMMAND
      1 sleep
$ time docker stop $cid
docker stop took: 10.23s
$ docker inspect $cid --format 'ExitCode={{.State.ExitCode}}'
ExitCode=137
```

`exec` place bien `sleep` en PID 1, mais l'arrêt prend les 10 s de timeout et se
termine par SIGKILL (137). **Ce n'est pas un défaut de l'image** : le noyau
n'applique pas les dispositions de signal par défaut à PID 1, et `sleep`
n'installe aucun gestionnaire SIGTERM. Aucun processus dans ce cas ne peut
répondre. Ce test ne prouve donc rien sur la propagation.

### 8b — contrôle : un PID 1 qui installe un gestionnaire

```console
$ cid=$(docker run -d --entrypoint bash agentic-gpu-worker:verify -c \
    'trap "echo caught-sigterm; exit 0" TERM; while true; do read -r -t 1 _ || true; done')
$ docker exec $cid ps -o pid,comm -p 1
    PID COMMAND
      1 bash
$ time docker stop $cid
docker stop took: 0.23s
$ docker logs $cid | tail -1
caught-sigterm
$ docker inspect $cid --format 'ExitCode={{.State.ExitCode}}'
ExitCode=0
```

Arrêt en **0,23 s**, exit 0. Les 10 s du cas 8a viennent donc bien de `sleep`
seul, et la livraison du signal à PID 1 fonctionne.

### 8c — test décisif : le **vrai** `entrypoint.sh`, avec un `vllm` de substitution

Aucun fichier du dépôt n'est modifié : un script `vllm` factice est monté depuis
l'hôte et placé en tête de `PATH`, si bien que `entrypoint.sh` exécute son
`exec "${vllm_argv[@]}"` normal et que c'est ce script qui est `exec`é.

```bash
# /tmp/.../fake/vllm
#!/usr/bin/env bash
echo "FAKE-VLLM: started, pid=$$, argv=$*"
trap 'echo "FAKE-VLLM: caught SIGTERM at pid $$, exiting cleanly"; exit 0' TERM
while true; do read -r -t 1 _ || true; done
```

```console
$ cid=$(docker run -d -v /tmp/.../vol8:/runpod-volume -v /tmp/.../fake:/fake:ro \
    -e WORKER_SKIP_MODEL_PREPARATION=1 \
    -e PATH=/fake:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    agentic-gpu-worker:verify)

$ docker exec $cid ps -o pid,ppid,comm,args -p 1
    PID    PPID COMMAND         COMMAND
      1       0 bash            bash /fake/vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --host 0.0.0.0 --port 8000 --served-model-name Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --max-model-len 16384 --gpu-memory-utilization 0.9 --tensor-parallel-size 1

$ docker logs $cid | tail -3
2026-09-18T13:19:38Z  STAGE ══ EXEC ══
2026-09-18T13:19:38Z  INFO  handing over to vLLM (pid 1 becomes PID 1 of the container)
FAKE-VLLM: started, pid=1, argv=serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --host 0.0.0.0 ...

$ time docker stop $cid
docker stop took: 0.21s

$ docker logs $cid | tail -1
FAKE-VLLM: caught SIGTERM at pid 1, exiting cleanly

$ docker inspect $cid --format 'ExitCode={{.State.ExitCode}}'
ExitCode=0
```

Ce test prouve **quatre** choses d'un coup :

1. `exec` remplace bien le shell : le processus final est **PID 1, PPID 0**.
   Aucun `bash` intermédiaire ne subsiste — le shell `entrypoint.sh` a disparu.
2. Le vecteur d'arguments transmis est exactement celui produit par
   `worker-serve-args` (visible dans `args` de `ps`).
3. SIGTERM atteint directement ce processus : le gestionnaire s'exécute et le
   message apparaît dans `docker logs`.
4. `docker stop` rend la main en **0,21 s**, exit **0**, sans SIGKILL.

**Ce qui reste non testé** : que le vrai binaire vLLM installe un gestionnaire
SIGTERM et termine proprement ses requêtes en vol, NCCL et les segments
`/dev/shm`. C'est une propriété de vLLM, pas de cette image, et elle exige un
GPU. Ce qui relève de l'image — placer le bon processus en PID 1 pour que le
signal lui parvienne sans intermédiaire — est prouvé.

---

## Synthèse

### Ce qui est prouvé

- L'image de base `vllm/vllm-openai:v0.28.0-cu129` se télécharge et contient
  vLLM 0.28.0+cu129, torch 2.13.0+cu129, transformers 5.15.1, hf-hub 1.28.0.
- Les couches du worker s'installent correctement et n'ajoutent que **2,2 MB**
  (étapes `[2/7]` à `[4/7]` du build committé, qui passent).
- **Aucun poids de modèle** n'est présent dans l'image : vérifié de l'extérieur
  par export et analyse des 141 553 entrées ; le plus gros fichier de type
  « poids » fait 1,4 MB et appartient à `compressed_tensors`.
- Les **huit** variables de cache sont dans l'ENV de l'image et pointent toutes
  sous `/runpod-volume` ; `Cmd` est `null` ; `start-period` = 45 min.
- Les six commandes console et les six scripts `docker/` sont installés,
  exécutables, et répondent.
- Les codes de sortie contractuels **3, 2, 2** sont respectés, avec des messages
  qui nomment la variable, la cause et l'action.
- Le protocole d'argv **NUL** résiste à un mot contenant un retour à la ligne,
  **dans le conteneur** : 17 mots lus = 17 mots produits par `vllm_argv()`,
  17 NULs, `\n` préservé à l'intérieur du dernier mot.
- Le cycle de vie complet se déroule, **sept étapes franchies**, jusqu'au `exec`
  de vLLM ; l'arborescence du volume est créée et réellement utilisée.
- `exec` place le processus final en **PID 1 / PPID 0**, SIGTERM lui parvient
  directement, `docker stop` rend la main en **0,21 s** avec exit 0.

### Ce qui n'est pas testable ici, et pourquoi

| Sujet | Raison |
|---|---|
| Démarrage effectif de vLLM, chargement FP8, capture de graphes CUDA | **Pas de GPU** sur la machine ; vLLM s'arrête sur `Failed to infer device type` |
| `worker-ready`, `worker-smoke-test`, une vraie complétion | Exigent un serveur vLLM opérationnel, donc un GPU |
| Le `HEALTHCHECK` en conditions nominales (réponse 200 sur `/health`) | Idem ; seul le chemin « rien n'écoute → exit 5 » a pu être vérifié |
| Arrêt gracieux **du vrai vLLM** (requêtes en vol, NCCL, `/dev/shm`) | Propriété de vLLM sous GPU ; seule la livraison du signal à PID 1 est prouvée (8c) |
| `worker-prepare-model` : téléchargement, verrou, reprise, marqueur | Téléchargerait 31,2 GB depuis le Hub ; non exécuté. `--check-only` a été exercé (retourne « model is not prepared ») |
| Compatibilité pilote NVIDIA 535–570, couverture Hopper 9.0 | Ne peut être vérifiée que sur du matériel Hopper |
| Tout le chemin RunPod : attachement du volume réseau, template de Pod, `tools/runpod_deployer` | Exige un compte et des ressources RunPod ; hors périmètre de cette machine |

### Défauts trouvés

---

#### D1 — BLOQUANT : la garde « pas de poids » a un faux positif garanti

`Dockerfile` **ligne 138**. Le `find` attrape `hadamards.safetensors`, un fichier
de données de la bibliothèque `compressed_tensors` livré par l'image de base —
1 436 901 octets de matrices de Hadamard, pas des poids de modèle. **Le build
échoue systématiquement**, sur toute machine, dès que l'image de base est
utilisée.

Reproduction :

```console
$ cd /home/hugo/Documents/programGen/gpu-worker
$ docker build --no-cache -t agentic-gpu-worker:verify .
...
#12 0.712 model weights in a layer: /usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
#12 0.712 + exit 1
ERROR: failed to build: ... did not complete successfully: exit code: 1
```

Confirmation que le fichier vient bien de l'image de base, et qu'il est seul :

```console
$ docker run --rm --entrypoint bash vllm/vllm-openai:v0.28.0-cu129 -c \
    'find / -xdev \( -name "*.safetensors" -o -name "*.gguf" \) -printf "%s\t%p\n" | sort -rn'
1436901	/usr/local/lib/python3.12/dist-packages/compressed_tensors/transform/utils/hadamards.safetensors
```

C'est le **seul** fichier correspondant dans toute l'image de base.

---

#### D2 — BLOQUANT : le build crée lui-même le `/runpod-volume` qu'il interdit

`Dockerfile` **ligne 140** (l'assertion) et **ligne 125** (la cause). L'`ENV`
des lignes 97-105 fixe `XDG_CACHE_HOME=/runpod-volume/xdg` **avant** le
`RUN pip install`. `pip` y écrit son cache HTTP — **malgré `--no-cache-dir`** —
et le nettoyage `rm -rf /root/.cache/pip` vise le mauvais chemin, puisque c'est
précisément `XDG_CACHE_HOME` qui a redirigé ce cache ailleurs.

Ce défaut est **indépendant de D1** : il se manifeste dès que D1 est neutralisé.

Reproduction (D1 neutralisé, D2 seul en cause) :

```console
$ docker build --no-cache -f Dockerfile.verify -t agentic-gpu-worker:verify .
#11 0.828 + found=
#11 0.828 + [ -n  ]
#11 0.828 + [ -e /runpod-volume ]
#11 0.828 + echo /runpod-volume must not exist in the image
#11 0.828 /runpod-volume must not exist in the image
#11 0.828 + exit 1
```

Cause racine isolée par exécution :

```console
$ docker run --rm -v .../pyproject.toml:/opt/worker/pyproject.toml:ro \
    -v .../README.md:/opt/worker/README.md:ro -v .../src:/opt/worker/src:ro \
    -e XDG_CACHE_HOME=/runpod-volume/xdg -e HF_HOME=/runpod-volume/huggingface \
    -e TMPDIR=/runpod-volume/tmp --entrypoint bash vllm/vllm-openai:v0.28.0-cu129 -c \
    'pip install --no-cache-dir --no-deps /opt/worker >/dev/null 2>&1; find /runpod-volume | head'

/runpod-volume/xdg
/runpod-volume/xdg/pip
/runpod-volume/xdg/pip/http-v2
/runpod-volume/xdg/pip/http-v2/3/3/9/7/4/33974f84394d9a943f68359da08431dab4af9f86c33962982ea21b5f.body
...
```

Le simple `import huggingface_hub` n'est **pas** en cause (vérifié séparément :
il ne crée rien) ; c'est bien `pip`.

**Gravité réelle, démontrée par exécution.** Si la garde de la ligne 140 était
supprimée au lieu d'être corrigée, l'image **livrerait** le point de montage,
qui est exactement le scénario catastrophe décrit dans les commentaires du
Dockerfile :

```console
$ sed '136,140d' Dockerfile > Dockerfile.noguard   # garde entièrement retirée
$ docker build --no-cache -f Dockerfile.noguard -t wv-noguard:tmp .   # rc=0
$ docker run --rm --entrypoint bash wv-noguard:tmp -c 'ls -la /runpod-volume; du -sh /runpod-volume; find /runpod-volume -type f | wc -l'
total 12
drwxr-xr-x 3 root root 4096 Sep 18 13:16 .
drwxr-xr-x 1 root root 4096 Sep 18 13:16 ..
drwxr-xr-x 3 root root 4096 Sep 18 13:16 xdg
920K	/runpod-volume
36
```

**920 KB et 36 fichiers de cache pip livrés dans l'image, au point de montage
du volume.** La garde fait donc correctement son travail : le Dockerfile viole
réellement son propre invariant. Corriger D2 signifie supprimer la cause (le
cache écrit sous `/runpod-volume`), pas la garde.

---

#### D3 — MINEUR : `TMPDIR` gravé pointe sur un répertoire inexistant hors du cycle de vie

`Dockerfile` ligne 103 : `TMPDIR=/runpod-volume/tmp`. Ce répertoire n'est créé
qu'à l'étape `VOLUME` d'`entrypoint.sh`. Tout processus lancé **sans passer par
l'entrypoint** — `docker run --entrypoint …`, `docker exec <pod> …` avant que le
volume ne soit prêt — hérite d'un `TMPDIR` cassé. Les outils coreutils échouent
sèchement ; Python retombe silencieusement sur `/tmp`.

Reproduction :

```console
$ docker run --rm --entrypoint bash agentic-gpu-worker:verify -c 'mktemp'
mktemp: failed to create file via template '/runpod-volume/tmp/tmp.XXXXXXXXXX': No such file or directory
```

Impact limité : le chemin nominal n'est pas affecté (l'étape `VOLUME` crée le
répertoire et ré-exporte `TMPDIR` avant que `entrypoint.sh` n'appelle lui-même
`mktemp` à l'étape `vLLM START`). C'est un piège pour l'opérateur et pour les
tests, pas pour la production. Il a d'ailleurs fait échouer la commande de
vérification du point 6 telle qu'elle était initialement rédigée.

---

## Reproduire cette vérification

```bash
cd /home/hugo/Documents/programGen/gpu-worker

# 0. image de base (~40 min à 5 MB/s, 24,2 GB décompressés)
docker pull vllm/vllm-openai:v0.28.0-cu129

# 1. le build committé — échoue (D1)
docker build --no-cache -t agentic-gpu-worker:verify .

# 1bis. copie de substitution, deux lignes, hors du dépôt
sed -e 's#rm -rf /root/.cache/pip; \\#rm -rf /root/.cache/pip /runpod-volume; \\#' \
    -e "s#-o -name '\*.gguf' \\\\) -print -quit#-o -name '*.gguf' \\\\) ! -path '/usr/local/lib/python3.12/dist-packages/compressed_tensors/*' -print -quit#" \
    Dockerfile > /tmp/Dockerfile.verify
docker build --no-cache -f /tmp/Dockerfile.verify -t agentic-gpu-worker:verify .

# 2. invariant « pas de poids », vérifié de l'extérieur
cid=$(docker create agentic-gpu-worker:verify)
docker export "$cid" | tar -tvf - > /tmp/export-listing.txt
docker rm -f "$cid"
awk '$1 ~ /^-/ {n=""; for(i=6;i<=NF;i++) n=n $i " "; printf "%12d  %s\n", $3, n}' \
    /tmp/export-listing.txt | sort -rn | head -10
grep -c runpod-volume /tmp/export-listing.txt   # attendu : 0

# 3. variables de cache
docker inspect agentic-gpu-worker:verify --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E '^(HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|VLLM_CACHE_ROOT|TORCH_HOME|TMPDIR|XDG_CACHE_HOME|TRITON_CACHE_DIR)='

# 4. commandes console
docker run --rm --entrypoint worker-preflight  agentic-gpu-worker:verify --skip-storage
docker run --rm --entrypoint worker-serve-args agentic-gpu-worker:verify --lines
docker run --rm --entrypoint worker-health     agentic-gpu-worker:verify   # attendu : 5

# 5. codes de sortie — ATTENTION : sous zsh, ne pas passer les -e via une variable
docker run --rm                                     agentic-gpu-worker:verify; echo $?  # 3
docker run --rm -e MAX_MODEL_LEN=10                 agentic-gpu-worker:verify; echo $?  # 2
docker run --rm -e MODEL_DOWNLOAD_MAX_ATTEMPTS=0    agentic-gpu-worker:verify; echo $?  # 2

# 6. protocole NUL (TMPDIR=/tmp requis, cf. D3)
# 7. cycle de vie : voir la section 7
# 8. SIGTERM : voir la section 8c
```

### Nettoyage effectué

Conteneurs de test supprimés, images intermédiaires (`wv-noguard:tmp`,
`agentic-gpu-worker:committed-attempt`) supprimées, `docker image prune`
exécuté. Restent volontairement en place :

```console
$ docker images | grep -E 'vllm|agentic-gpu-worker'
agentic-gpu-worker:verify        38ca731a5074       24.2GB             0B
vllm/vllm-openai:v0.28.0-cu129   249ed60fdd67       24.2GB             0B
```

Rappel : `agentic-gpu-worker:verify` est l'**image de substitution** (deux lignes
corrigées), pas le produit du `Dockerfile` committé — lequel ne produit aucune
image à ce jour.
