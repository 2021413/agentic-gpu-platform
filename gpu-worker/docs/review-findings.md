# Revue adverse — gpu-worker

Revue contradictoire du livrable, menée sans GPU, sans compte RunPod et sans
`docker build` (interdit : une autre session construit en parallèle).

Tout ce qui suit est appuyé soit par un `fichier:ligne`, soit par une commande
réellement exécutée dont la sortie est reproduite.

Commandes de référence utilisées :

```bash
cd /home/hugo/Documents/programGen/gpu-worker
.venv/bin/python -m pytest tests -q          # 335 passed, 1 skipped
.venv/bin/python -m ruff check .             # All checks passed!
.venv/bin/python -m mypy                     # Success: no issues found in 12 source files
.venv/bin/python -m pytest tests -q --cov=src --cov=tools --cov-report=term-missing
```

---

## 1. Tableau des 20 critères

| # | Critère | Verdict | Preuve / ce qui manque |
|---|---|---|---|
| 1 | L'image Docker se construit | NON VÉRIFIABLE ICI | `docker build` interdit. L'étape la plus fragile (`Dockerfile:114-130`, build du wheel) a été reproduite hors Docker avec exactement le contexte que `.dockerignore` laisse passer (`pyproject.toml`, `README.md`, `src/`, `docker/`) : `uv build --wheel` → `Successfully built agentic_gpu_worker-1.0.0-py3-none-any.whl`, 6 console-scripts présents. Reste non prouvé : pull de l'image de base, `command -v bash`, et le `RUN` de garde `Dockerfile:137-140`. |
| 2 | Aucun poids de modèle dans les couches | TENU PAR CONSTRUCTION, NON PROUVÉ PAR UN TEST | Aucun `COPY`/`ADD`/téléchargement de poids ; `.dockerignore:74-79` ; garde `Dockerfile:137-140`. Mais cette garde ne s'exécute que pendant `docker build`, et **rien dans le dépôt ne lance de build** (pas de CI — voir F3). `README.md:146-148` affirme le contraire. La garde ne cherche que `*.safetensors` et `*.gguf` (pas `*.bin`, `*.pt`). |
| 3 | Données volumineuses sur le montage persistant | TENU | `Dockerfile:88-96` + `src/worker/config.py:146-162` + `docker/entrypoint.sh:144-180`. Test : `tests/unit/test_config.py:482`. Réserves : les 8 `ENV` du Dockerfile sont recopiées à la main et **aucun test ne les compare** à `PersistentLayout.environment()` ; `layout.models` et `layout.logs` sont créés et jamais utilisés. |
| 4 | Absence de stockage persistant → échec rapide | TENU | `src/worker/filesystem.py:174-212` (inexistence, mount partagé avec `/`, non-inscriptible), `src/worker/cli.py:166-170` → code 3. Tests : `tests/unit/test_filesystem.py:31,45,68,83`, `tests/unit/test_cli.py:146`. C'est la première étape du boot (`docker/entrypoint.sh:131`). |
| 5 | Stockage insuffisant → diagnostic utile | PARTIELLEMENT TENU | Le diagnostic existe et est testé (`src/worker/model_state.py:377-390`, `src/worker/cli.py:181-186`, `tests/unit/test_cli.py:158`). **Mais le garde-fou rapide est contournable** : `cli.py:181` teste `marker is None` au lieu de « marqueur correspondant », donc un marqueur d'un *autre* modèle fait passer preflight avec 0 Go utile (voir F8, reproduit). |
| 6 | Les téléchargements interrompus reprennent | PARTIELLEMENT TENU | La reprise est celle de `huggingface_hub` (`.incomplete` + en-tête `Range`, vérifié dans `.venv/.../huggingface_hub/file_download.py:377-378,1282`) et rien dans ce dépôt ne supprime les fichiers partiels. **Aucun test ne prouve la reprise** : les tests unitaires prouvent seulement que le downloader est rappelé (`tests/unit/test_model_state.py:400-418`), et le seul test qui prétend prouver la réparation est tautologique (F5). Pire : un fichier *tronqué* n'est jamais réparé (F1). |
| 7 | Bootstrap concurrent ne peut pas doubler le téléchargement | TENU | `src/worker/model_state.py:168-216` (flock) + re-vérification sous verrou `model_state.py:320-325`. Tests solides avec **deux vrais processus** : `tests/unit/test_model_state.py:614-626` (bloqué), `:628-653` (verrou libéré au SIGKILL), `:708-737` (le perdant ne télécharge pas). |
| 8 | La révision du modèle peut être épinglée | TENU | `config.py:336`, `config.py:313-316` (`--revision` seulement sans snapshot local), `model_state.py:334-336`, `model_state.py:98-107`. Tests : `test_config.py:336`, `test_model_state.py:362-377`. Réserves : le marqueur enregistre la **chaîne demandée**, pas le commit résolu, dès que `MODEL_REVISION` est posé (F13) ; l'émission de `MODEL_REVISION` par le déployeur n'est couverte par aucun test (`models.py:207-208`, ligne manquante au coverage). |
| 9 | vLLM démarre en utilisant le modèle préparé | NON VÉRIFIABLE ICI | Pas de GPU, pas de vLLM installé (`worker-preflight` affiche `vllm not installed`). La logique « pointer vLLM sur le snapshot vérifié » est testée en unitaire : `tests/unit/test_cli.py:357-370`. Pour vérifier : un Pod H100 réel, ou au minimum `docker run` avec `WORKER_SKIP_MODEL_PREPARATION=1`. |
| 10 | `/v1/models` répond | NON VÉRIFIABLE ICI | Seulement contre `respx` (`tests/unit/test_readiness.py:65-120`). Le seul test réel est `tests/integration/test_runpod_real.py:162`, jamais exécuté (skip : `RUNPOD_API_KEY is not set`). |
| 11 | Le smoke test réalise une inférence valide | NON VÉRIFIABLE ICI | Validation de forme correcte et bien testée en mock (`tests/unit/test_readiness.py:308-465`, y compris le 200 vide). L'inférence réelle n'existe que dans le test RunPod jamais lancé. |
| 12 | SIGTERM arrête vLLM proprement | NON VÉRIFIABLE ICI | Unique preuve : `exec` à `docker/entrypoint.sh:306` (correct, et argumenté `:289-305`). Aucun test, aucun `shellcheck`, aucun `bats` : **la couche shell n'a zéro couverture**. Pour vérifier : `docker run` + `docker stop` en mesurant le code de sortie 143 et l'absence de SIGKILL. |
| 13 | Redémarrage sur le même volume sans re-téléchargement | TENU (niveau bibliothèque) | Chemin chaud : `model_state.py:307-309,353-374`. Tests : `tests/unit/test_model_state.py:262-278` (le downloader lève si appelé), `tests/integration/test_real_model_preparation.py:73-85` contre le vrai `huggingface_hub`. Bout-en-bout seulement dans le test RunPod jamais lancé. |
| 14 | H100 mono-GPU fonctionne | NON VÉRIFIABLE ICI | Aucun GPU. `tests/integration/test_runpod_real.py:55` prévoit `NVIDIA H100 80GB HBM3`, mais le test est skip. |
| 15 | H200 est supporté | NON VÉRIFIABLE ICI, ET NON ÉTAYÉ | H200 n'apparaît que dans de la prose (`docs/runpod.md:10,161`, `.env.example:148`, `Dockerfile:28`, `docs/troubleshooting.md:89`). Aucun identifiant de type GPU H200 nulle part, aucun paramétrage de test, aucun artefact de mesure. La seule justification technique est « compute 9.0 ≥ 8.9 » (`src/worker/gpu.py:331-337`). |
| 16 | Le tensor parallel est configurable | TENU pour `TENSOR_PARALLEL_SIZE` / NON TENU pour `AUTO_TENSOR_PARALLEL` | `config.py:344`, `config.py:310-311`, test `test_config.py:304`. Mais `resolve_tensor_parallel` (`config.py:453-464`) n'est appelée **par aucun code de production** — uniquement par `tests/unit/test_config.py:441-458`. Voir F6. |
| 17 | La CI normale passe sans GPU | NON TENU | **Aucune CI n'existe** : pas de `.github/`, pas de `Makefile`, pas de `.gitlab-ci.yml` (`find` sur l'arbre complet). De plus, la commande documentée `pytest tests -q` **exécute 4 tests réseau** qui téléchargent depuis huggingface.co (voir F4). |
| 18 | Les credentials RunPod ne fuient jamais | TENU côté déployeur, NON TENU côté worker | Déployeur : `client.py:107-124` (`redact`), `models.py:76-107` (`SecretValue`), tests `test_runpod_cli.py:96-115`, `test_runpod_client.py`. Worker : `VLLM_API_KEY` est écrit **en clair dans un fichier du volume persistant qui n'est jamais supprimé** (F2). |
| 19 | Les tags d'image de production sont épinglés | PARTIELLEMENT TENU | Image de base épinglée par **tag** et non par digest (`Dockerfile:37`). Côté déploiement, `--image` / `WORKER_IMAGE` accepte n'importe quoi, `:latest` compris — aucune validation (`tools/runpod_deployer/cli.py:469-474`), aucun test de refus. `README.md:166-170` promet des tags `<semver>`/`<git-sha>` ; rien ne les produit ni ne les vérifie. |
| 20 | La doc de déploiement suffit à un autre ingénieur | PARTIELLEMENT TENU | `docs/runpod.md` est bon pour un déploiement console (GPU, volume, ports, proxy vs TCP). Mais : **`tools/runpod_deployer` (1 400 lignes, le chemin de déploiement automatisé) n'est mentionné dans aucun document** (`grep runpod_deployer docs/ README.md` → rien) ; `WORKER_IMAGE` et `RUNPOD_API_KEY` ne sont documentés nulle part hors `--help` ; et plusieurs affirmations des docs sont fausses (F3, F6, F7, F12, F13, F14). |

### Exigences transverses

| Exigence | Verdict | Preuve |
|---|---|---|
| Aucun `eval` de chaîne d'environnement | TENU | `grep -rn '\beval\b'` → seulement deux commentaires (`entrypoint.sh:142`, `preflight.sh:25`). Les variables sont lues `key=value` et exportées une par une (`entrypoint.sh:154-160`). |
| `set -Eeuo pipefail` dans chaque script shell | TENU | 6/6 : `entrypoint.sh:16`, `health.sh:16`, `preflight.sh:16`, `prepare-model.sh:18`, `ready.sh:14`, `smoke-test.sh:13`. (Les `RUN` du Dockerfile utilisent `set -eux` sous `sh` : pas de `pipefail`, mais ce ne sont pas des scripts.) |
| Un seul processus vLLM, lancé en `exec` | TENU PAR INSPECTION | `docker/entrypoint.sh:306`. Aucun test. Un processus supplémentaire existe si `WORKER_WAIT_READY=1` (`entrypoint.sh:277-284`), documenté honnêtement. |
| Secrets jamais journalisés, jamais dans les couches ni les build args | NON TENU | Build args : OK (`Dockerfile:46-56`, aucun secret). stdout/stderr : OK et testé (`tests/unit/test_cli.py:603-657`). **Mais** `VLLM_API_KEY` atteint un fichier du volume (F2) et la ligne de commande de vLLM (`/proc/1/cmdline`, `ps`). |
| Pas de poids téléchargés pendant le build | TENU | Aucun appel réseau dans les `RUN` ; `pip install --no-deps /opt/worker` (`Dockerfile:124`). |
| Marqueur de préparation atomique | TENU | `model_state.py:148-162` (écriture sur un frère + `os.replace`). Tests : `test_model_state.py:89-99`. |
| Échecs de quota non retentés indéfiniment | TENU | `model_state.py:424-433` (aucun retry, aucun backoff) + `filesystem.py:278-294` (y compris via `__cause__`). Tests : `test_model_state.py:447-466`, `:485-496`. |
| Verrou de système de fichiers | TENU | `flock` `model_state.py:190`, preuve multi-processus `test_model_state.py:614-653`. |
| Tentatives bornées | TENU | `model_state.py:409` + refus de `MODEL_DOWNLOAD_MAX_ATTEMPTS=0` à la construction (`config.py:224-253`, test `test_config.py:508`). Côté déployeur : `RetryPolicy` (`client.py:205-214`), 401/403 jamais retentés (`client.py:87`). |

---

## 2. Ce qui est faux, manquant ou trompeur — par gravité

### F1 — CRITIQUE — Un fichier tronqué sur le volume n'est jamais réparé, et le marqueur blanchit la corruption

`src/worker/model_state.py:331-343`. Après un téléchargement, `snapshot_files(snapshot)`
**re-mesure les tailles actuelles** et les écrit dans le marqueur avec `verified=True`.
Or `snapshot_download` ne re-télécharge pas un blob déjà présent dont l'etag correspond,
même s'il est court. Résultat : un shard tronqué est détecté une fois, « re-préparé »
sans rien réparer, puis **enregistré à sa taille tronquée**. Tous les boots suivants
vérifient « OK ».

Reproduction (exécutée, sortie réelle) :

```
$ .venv/bin/python  # télécharge hf-internal-testing/tiny-random-gpt2, tronque model.safetensors, relance prepare_model
weights model.safetensors 453864 -> 226932
after prepare: 226932 STILL TRUNCATED
# et sur tokenizer_config.json :
after 'repair': 118   downloaded flag: True
marker size for victim: 118
verify_snapshot on the new marker: True []
```

Le cas *fichier manquant* est bien réparé (`RESTORED 453864`) ; c'est uniquement le cas
*tronqué* qui échoue — exactement celui que la conception revendique
(`model_state.py:235-241` : « une taille change quand un shard est tronqué »).

Contredit : `README.md:73-75`, `docs/persistent-storage.md:86-88`,
`docs/troubleshooting.md:58-61` (« Le téléchargement reprend ; rien n'est supprimé »).

Conséquence : vLLM démarre sur des poids corrompus, et le volume ment pour toujours.

### F2 — CRITIQUE — `VLLM_API_KEY` est écrit en clair sur le volume persistant et n'est jamais effacé

`docker/entrypoint.sh:234-237` :

```bash
serve_args_file="$(mktemp)"
trap 'rm -f "${serve_args_file}"' EXIT
worker-serve-args > "${serve_args_file}"
```

Trois faits, chacun vérifié :

1. `worker-serve-args` **sans** `--redacted` émet la clé en clair
   (`src/worker/config.py:317-318`). Exécuté :
   `vllm serve acme/x ... --api-key super-secret-key-123`.
2. `TMPDIR` vaut `/runpod-volume/tmp` (`Dockerfile:94`, ré-exporté `entrypoint.sh:154-179`)
   et `mktemp` honore `TMPDIR` (vérifié). Le fichier atterrit donc **sur le volume réseau
   partagé**, pas sur le disque éphémère.
3. `exec` (`entrypoint.sh:306`) remplace le shell : **le trap `EXIT` ne s'exécute jamais**.
   Vérifié : `bash -c 'trap "echo TRAP-RAN" EXIT; exec /bin/true'` n'affiche rien.

Donc chaque démarrage de Pod dépose un fichier `/runpod-volume/tmp/tmp.XXXXXXXXXX`
contenant le jeton d'API en clair, qui survit au Pod, s'accumule, et est lisible par tout
Pod montant le même volume (tous tournent en root).

Le test « aucun secret ne fuit » (`tests/unit/test_cli.py:603-637`) ne couvre que
stdout/stderr, donc il passe.

Corollaire de moindre gravité mais réel : la clé est aussi dans `argv` de vLLM, donc dans
`/proc/1/cmdline` et `ps` à l'intérieur du conteneur. vLLM lit nativement `VLLM_API_KEY`
depuis l'environnement ; le passer en `--api-key` est un choix qui coûte cette exposition.

### F3 — MAJEUR — Il n'y a aucune CI, et le README affirme qu'il y en a une qui prouve l'invariant

`README.md:146-148` : « CI never needs a GPU and never downloads the model. It also
asserts the invariant mechanically: the built image is exported and searched for weight
files, and the boot refuses without a volume. »

Aucun de ces mécanismes n'existe : pas de `.github/`, pas de `Makefile`, pas de fichier de
pipeline (arbre complet listé ; `.dockerignore:20,53` exclut `.github` et `Makefile`, qui
n'existent pas). Rien n'exporte l'image, rien ne la scanne, rien ne teste un boot sans
volume au niveau conteneur. Le critère 17 repose donc entièrement sur un `pytest` lancé à
la main.

### F4 — MAJEUR — `pytest tests -q` (la commande documentée comme « no network ») télécharge depuis huggingface.co

`README.md:140-143` annonce `pytest tests -q  # no GPU, no network, no container`.
Or `pyproject.toml:73-82` ne contient **aucun `addopts`** désélectionnant `integration`,
et `tests/integration/test_real_model_preparation.py:24` n'est marqué que `integration`.

Exécuté :

```
$ .venv/bin/python -m pytest tests/integration/test_real_model_preparation.py -v
test_a_cold_start_downloads_verifies_and_marks PASSED
test_a_warm_restart_reuses_the_volume_without_downloading PASSED
test_a_truncated_shard_is_detected_and_completed PASSED
test_the_marker_survives_a_process_restart PASSED
4 passed in 8.01s
```

Ces 4 tests font partie des 335 de la passe « normale ». Sur une machine sans réseau ils
se *skippent* proprement (`:39-42`), donc la suite n'échoue pas — mais l'affirmation
« no network » est fausse, et une CI hors-ligne perdrait silencieusement la seule
couverture contre le vrai `huggingface_hub`.
Deuxième inexactitude de la même section : `-m integration` est décrit comme « needs a
container runtime » alors qu'aucun test du dépôt ne démarre de conteneur.

### F5 — MAJEUR — Le test qui prétend prouver la réparation d'un shard tronqué est tautologique

`tests/integration/test_real_model_preparation.py:88-110`, nommé
`test_a_truncated_shard_is_detected_and_completed`, se termine par :

```python
repaired = prepare_model(config, log=lambda _m: None)
assert repaired.downloaded is True
ok, problems = verify_snapshot(Path(repaired.state.snapshot_path), repaired.state.files)
assert ok, problems
```

`repaired.state.files` vient d'être produit par `snapshot_files()` sur ces mêmes fichiers
(`model_state.py:331`). L'assertion compare donc le disque à lui-même : elle est vraie
quel que soit l'état réel. Le test ne vérifie jamais que la taille d'origine est revenue —
et F1 montre qu'elle ne revient pas. Le nom et la docstring (« must notice », « completed »)
affirment davantage que ce qui est prouvé.

### F6 — MAJEUR — `AUTO_TENSOR_PARALLEL` est du code mort ; deux documents affirment qu'il fonctionne

`src/worker/config.py:453-464` définit `resolve_tensor_parallel`. Recherche exhaustive :
elle n'est appelée que par `tests/unit/test_config.py:441-458`. `vllm_argv`
(`config.py:310-311`) utilise `self.tensor_parallel_size` tel quel.

- `README.md:104` : « `AUTO_TENSOR_PARALLEL=true` uses every visible GPU. » → **faux**.
- `docs/runpod.md:161-163` : « `AUTO_TENSOR_PARALLEL=true` opts into using every visible
  GPU. » → **faux**.
- `.env.example:152-157` dit au contraire la vérité (« the current code path records this
  flag and shows it in the boot banner, but the vLLM argument vector is built from
  TENSOR_PARALLEL_SIZE verbatim »).

Le déployeur propage tout de même la variable jusqu'au Pod (`models.py:211-212`), où elle
n'aura aucun effet — et cette ligne n'est couverte par aucun test (manquante au coverage).

### F7 — MAJEUR — La doc décrit un refus de sécurité qui n'existe pas dans le code

`docs/troubleshooting.md:142-145` :

> ### `a non-empty service token is required`
> `VLLM_API_KEY` is unset while the port is exposed publicly. Refusing to start beats
> starting with authentication silently disabled.

Ce message n'existe nulle part (`grep -rn "non-empty service token"` → seulement cette
ligne de doc). Le worker **ne refuse pas** de démarrer sans `VLLM_API_KEY` : il affiche
`api key NOT SET (open port)` (`config.py:370`) et continue. `.env.example:208-209` le dit
correctement. Un ingénieur qui fait confiance au troubleshooting croira être protégé par
un garde-fou inexistant sur un port public devant un GPU.

### F8 — MOYEN — Le contrôle d'espace libre de preflight est contourné dès qu'un marqueur d'un autre modèle traîne

`src/worker/cli.py:181` : `if marker is None and not report.has_enough_free:`.
`_report_model` (`cli.py:125-133`) renvoie le marqueur **sans vérifier `matches()`**.
Donc : volume contenant le modèle A, `MODEL_ID` changé pour B, volume plein → preflight
passe.

Reproduit (marqueur `acme/OLD-model`, `MODEL_ID=acme/NEW-model`, `MIN_FREE_DISK_GB=1000000`) :

```
  model cache in use     0.0 GB
  required free          1,000,000.0 GB
  acme/OLD-model@deadbeefcafe
preflight passed
EXIT=0
```

Le manque est rattrapé plus tard par `_require_free_space` (`model_state.py:377-390`),
donc ce n'est pas fatal — mais c'est précisément le scénario « upgrade de révision » que
`docs/persistent-storage.md:50` chiffre, et preflight existe pour échouer *avant*.
Aucun test ne couvre ce cas ; `tests/unit/test_cli.py:168` ne teste que le marqueur
correspondant.

### F9 — MOYEN — `WORKER_ALLOW_EPHEMERAL_STORAGE=on` : le shell l'accepte, Python le refuse

`docker/entrypoint.sh:117-118` accepte `1 | true | yes | on`.
`src/worker/cli.py:82` n'accepte que `("1", "true", "yes")`.

Avec `on`, le boot affiche les trois lignes d'avertissement « une absence de volume
n'arrêtera PAS ce worker » puis `worker-preflight` exige le montage et sort en code 3,
avec une explication (`entrypoint.sh:56-62`) qui contredit le message précédent.
Le reste du code utilise `_optional_flag` qui, lui, accepte `on/off` (`config.py:408-409`) :
deux conventions booléennes coexistent. `.env.example:76` ne documente que `=1`.

### F10 — MOYEN — `preflight.sh` et `prepare-model.sh` ignorent l'échec du Python qui redérive les caches

`docker/preflight.sh:26-33` et `docker/prepare-model.sh:27-34` lisent la sortie de
`python3 -c ...` via une **substitution de processus**, dont le code de retour n'est jamais
consulté — et avec `set -Eeuo pipefail` cela n'interrompt rien.

Vérifié :

```
$ bash -c 'set -Eeuo pipefail; trap "echo ERR" ERR; while read -r l; do :; done < <(python3 -c "import sys; sys.exit(3)"); echo "script continued"'
script continued
```

Si ce Python échoue (config invalide, OOM, interpréteur cassé), le wrapper poursuit avec
les valeurs *bakées* de l'image. Avec un `PERSISTENT_ROOT` personnalisé, `HF_HUB_CACHE`
reste `/runpod-volume/huggingface/hub` — c'est-à-dire exactement le résultat que le
commentaire de `prepare-model.sh:11-14` déclare vouloir empêcher (« Getting this wrong
writes 31.2 GB to the container filesystem instead of the volume »).
`entrypoint.sh:144-152` traite correctement ce cas ; les deux wrappers non.

### F11 — MOYEN — Tout l'outil de déploiement est absent de la documentation

`tools/runpod_deployer/` (client 849 l., cli 586 l., models 572 l.) fournit
`deploy/status/smoke-test/destroy/gpu-types`, gère le data center du volume, le choix
proxy/TCP, les codes de sortie 0-8. `grep -rn "runpod_deployer" README.md docs/` → **aucune
occurrence**. `WORKER_IMAGE` (`cli.py:65`) n'est documenté nulle part hors `--help`.
`docs/runpod.md` décrit uniquement un déploiement manuel par la console.
Le critère 20 (« suffit à un autre ingénieur ») est donc tenu pour le chemin console, et
manqué pour le chemin outillé.

### F12 — MOYEN — `docs/runpod.md` décrit un agent d'auto-enregistrement et un heartbeat qui n'existent pas

`docs/runpod.md:98-99` : « That is handled by design here — the worker agent re-registers
its endpoint on every boot ». `docs/runpod.md:172-175` : « The worker registers itself with
the control plane, starts heartbeating, and receives jobs. Remove one by draining it: it
stops accepting work, finishes what it holds, and deregisters. »

Rien de tel n'existe dans ce dépôt : aucun enregistrement, aucun heartbeat, aucun drain
(`grep -rin "heartbeat\|register\|drain" src/ docker/` → rien). Le worker est un serveur
vLLM passif. Un lecteur du guide de déploiement conclura que le changement de port TCP au
reset est géré ; il ne l'est pas.

### F13 — MOYEN — Le marqueur n'enregistre pas « le commit résolu » quand une révision est épinglée

`src/worker/model_state.py:335` : `revision=config.model_revision or _resolved_revision(snapshot)`.
Quand `MODEL_REVISION` est posé, la valeur stockée est **la chaîne demandée**, pas le
commit résolu. Avec `MODEL_REVISION=main`, le marqueur dit `revision: "main"` et
`matches()` (`model_state.py:98-107`) l'acceptera indéfiniment : l'épinglage sur une
branche mouvante ne redéclenche jamais rien tout en *ressemblant* à un épinglage.

Contredit `docs/persistent-storage.md:65-67` (« It records the model id, **the resolved
commit** ») et `docs/troubleshooting.md:69-70`. Aucun test ne couvre un `MODEL_REVISION`
non-SHA ; `tests/unit/test_model_state.py:362-377` n'utilise que `"abc123"` et vérifie
justement que la chaîne demandée est recopiée.

### F14 — MINEUR — `docs/persistent-storage.md:18` : les téléchargements partiels ne vont pas dans `tmp/`

La doc écrit « `tmp/` TMPDIR — partial downloads land here, not in the container ».
`huggingface_hub` écrit les partiels en `<HF_HUB_CACHE>/blobs/<etag>.incomplete`
(vérifié : `file_download.py:1282`, `incomplete_path=Path(blob_path + ".incomplete")`).
Les deux chemins sont sur le volume, donc sans conséquence opérationnelle, mais la marge
de reprise chiffrée dans le tableau de dimensionnement (`:46`) se consomme dans `hub/`,
pas dans `tmp/`. Le cache Xet (`HF_XET_CACHE` → `<HF_HOME>/xet`, vérifié) est également
absent de l'arborescence documentée `:10-24` et n'est compté par aucune mesure
(`directory_size_bytes` ne regarde que `hub_cache`, `filesystem.py:273`).

### F15 — MINEUR — Chemins de code que rien n'exécute

Mesuré par `pytest --cov=src --cov=tools --cov-report=term-missing` :

- `src/worker/cli.py:267-269` — la branche `--lines` de `serve_args_main`, **c'est
  pourtant exactement ce que `docker/entrypoint.sh:252-254` exécute** pour journaliser la
  commande vLLM. Zéro test.
- `tools/runpod_deployer/models.py:208,210,212,216` — émission de `MODEL_REVISION`,
  `SERVED_MODEL_NAME`, `AUTO_TENSOR_PARALLEL`, `HF_HUB_DISABLE_XET` vers le Pod : aucune
  couverture (donc l'épinglage de révision *par le déployeur* n'est prouvé nulle part).
- Fonctions jamais appelées en production, seulement par les tests :
  `config.py:453` `resolve_tensor_parallel`, `config.py:277` `hub_environment`
  (conséquence : `HF_HUB_DISABLE_XET` n'est jamais ré-exporté par le worker ; il ne
  fonctionne que parce que la variable brute est déjà dans l'environnement),
  `model_state.py:495` `clear_model_cache`, `model_state.py:510` `storage_error_hint`.
- Répertoires créés et jamais utilisés : `layout.models` et `layout.logs`
  (`config.py:90-91,117-119`), le premier documenté « reserved for manually placed
  weights ».

### F16 — MINEUR — `redacted_argv` ne masque que la clé qu'il a lui-même ajoutée

`src/worker/config.py:322-328` cherche `--api-key` dans le vecteur. Si l'opérateur écrit
`VLLM_EXTRA_ARGS='--api-key sk-...'` sans poser `VLLM_API_KEY`, `self.vllm_api_key` est
faux, aucun masquage n'a lieu, et `docker/entrypoint.sh:251-254` imprime la clé en clair
dans les logs du conteneur. `.env.example:104-109` annonce des extra-args « appended
verbatim » sans avertir de ce cas.

### F17 — MINEUR — La garde anti-poids du Dockerfile est plus étroite que le `.dockerignore`

`Dockerfile:138` ne cherche que `*.safetensors` et `*.gguf`, alors que
`.dockerignore:74-77` se protège aussi de `*.pt` et `*.bin`. Un modèle au format `.bin`
(ou un checkpoint `.pt`) passerait la garde. Le `find / -xdev` ne traverse pas non plus
les montages, ce qui est correct dans un build mais n'inspecte pas les couches de base
montées ailleurs.

### F18 — MINEUR — L'étape « MODEL VALIDATION » de l'entrypoint promet plus que ce qu'elle fait

`docker/entrypoint.sh:204-206` : « Cheap re-check of the marker: proves that what is on the
volume is what this worker was asked to serve ». La commande exécutée,
`worker-prepare-model --check-only` (`src/worker/cli.py:205-211`), ne lit que le marqueur
et appelle `matches()` : elle **ne vérifie aucun fichier** (pas de `verify_snapshot`).
Un marqueur valide pointant vers un snapshot supprimé passe cette étape, et
`serve_args_main` (`cli.py:258-263`) appliquera la même condition incomplète pour pointer
vLLM sur un chemin inexistant.

### F19 — MINEUR — Le worker n'a pas d'équivalent de `redact()` pour ses propres journaux

Le déployeur fait passer toute chaîne sortante par `redact` (`client.py:107-124`). Le
worker, lui, journalise le texte brut des exceptions du client hub
(`model_state.py:438`) et l'insère dans le message d'erreur final (`model_state.py:445`),
qui est imprimé par l'entrypoint. Aucune fuite n'a été démontrée ici (les exceptions de
`huggingface_hub` ne contiennent normalement pas le jeton), mais c'est le seul chemin du
worker où une chaîne venue du réseau atteint les logs sans filtre — asymétrie à noter face
au soin pris ailleurs.

### F20 — MINEUR — Incohérences de chiffres et d'exemples entre documents

- Volume recommandé : `README.md:33` et `docs/runpod.md:12` disent 150 Go ;
  `docs/persistent-storage.md:52-53` dit « Recommended 150 GB. Absolute minimum 80 GB »
  tout en chiffrant « Single revision, comfortable ~50 GB ». Trois nombres cohabitent.
- `docs/persistent-storage.md:69-79` montre un marqueur d'exemple sans le champ
  `vllm_image` que le code écrit systématiquement (`model_state.py:341`).
- `Dockerfile:56` fige `VALIDATED_MODEL_REVISION` dans un LABEL, tandis que
  `README.md:37` et `docs/runpod.md:27` demandent de poser la même valeur à la main :
  rien ne vérifie que les deux restent synchronisées.

---

## 3. Ce que j'ai cherché sans le trouver (donc : à mettre au crédit du livrable)

- **Injection de commande par l'environnement** : aucune. `shlex.split` au parse
  (`config.py:433-450`), vecteur `argv` NUL-séparé (`cli.py:270`), `mapfile -d ''`
  (`entrypoint.sh:244`), aucun `eval`. Test dédié : `test_config.py:221` (`; curl evil.sh | sh`
  devient des mots inertes) et `test_cli.py:418` (mot contenant un saut de ligne).
- **Secret dans un build arg ou une couche** : aucun (`Dockerfile:46-56`).
- **Course entre deux Pods** : réellement testée avec deux processus système, y compris la
  libération du verrou par SIGKILL — c'est la partie la mieux prouvée du livrable.
- **Retry sur disque plein** : correctement absent, testé sur trois formes d'erreur
  (ENOSPC, EDQUOT, message encapsulé via `__cause__`).
- **Assertions tautologiques** : cherchées dans les 336 tests ; une seule trouvée (F5).
  Les mocks `respx` valident bien les vraies fonctions de décodage, pas des doublures.
- **Typage / lint** : `ruff` et `mypy --strict` passent proprement sur `src` et `tools`.
- **Variables lues mais non documentées** : aucune dans `src/worker/` (les 25 variables de
  `.env.example` correspondent à `config.py` + `entrypoint.sh`). Les seules non documentées
  sont celles des outils (`WORKER_IMAGE`, `RUNPOD_TEST_*`) — voir F11.
- **Messages d'erreur nommant une variable ou un chemin disparu** : aucun trouvé dans le
  code ; le seul cas est dans la documentation (F7).
