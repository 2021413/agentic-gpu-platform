# État du projet — problèmes restants

Rédigé le 22 septembre 2026, après la première série de runs contre un vrai
modèle (`Qwen3-Coder-30B-A3B-Instruct-FP8` servi par vLLM sur une H200 RunPod).

Ce document ne liste que ce qui **reste**. Ce qui a été corrigé est dans
l'historique git ; ce qui compte ici, c'est ce qui peut encore faire échouer un
run et ce que je n'ai pas pu prouver.

Une règle de lecture : je distingue partout **vérifié** (exécuté, avec la
sortie sous les yeux) de **supposé** (lu dans le code, jamais éprouvé). Cette
distinction est la seule qui ait vraiment coûté cher dans ce projet.

---

## 1. Ce qui n'a jamais rencontré un vrai modèle

Six corrections de la boucle de réparation ont été livrées le 22 septembre.
Elles sont prouvées par mutation sur le chemin déterministe — je retire la
correction, le test tombe — et **aucune n'a tourné contre un GPU**, parce que
EU-FR-1 est passée à zéro disponibilité entre le test et la livraison.

Ce qui est donc inconnu :

- Le modèle **lit-il** la sortie d'outils qu'il reçoit désormais, ou la
  survole-t-il ? Le prompt lui dit de la lire d'abord ; un modèle de 30B peut
  ignorer une instruction.
- Le brief cumulé marque les objections répétées (`[raised 3 times; still not
  fixed]`). Personne n'a vérifié que ça change son comportement.
- Le reviewer est maintenant prié de localiser ses reproches. Le schéma laisse
  `file` et `line` **facultatifs** (`str | None`, `int | None`), donc rien ne
  l'y oblige mécaniquement. Un modèle qui les omet produira toujours des
  briefs vagues.

**Comment lever le doute :** `./scripts/retry-gpu-test.sh`, puis comparer le
nombre de tours de réparation et le taux de findings localisés avec les
mesures du 22 septembre (3 tours, 6 findings sur 6 sans localisation).

**Rendre `file` obligatoire** serait la correction dure, mais elle ferait
échouer la sortie structurée quand le modèle ne sait honnêtement pas où
pointer, ce qui coûte un tour complet. Choix délibérément laissé ouvert.

---

## 2. Le budget de réparation est peut-être le mauvais chiffre

```
max_repair_iterations = 3
max_coder_iterations  = 6
```

Les trois runs réels ont tous fini par `repair budget exhausted`. À l'époque,
le coder ne recevait aucune information, donc trois tours ne pouvaient rien
donner — le chiffre n'était pas le problème.

Maintenant qu'il reçoit la sortie des tests, trois tours est peut-être trop
peu, ou parfaitement suffisant. **Aucune donnée.** Ne pas toucher ce chiffre
avant d'avoir mesuré : l'augmenter multiplie directement la facture GPU.

---

## 3. Pas de streaming — et le plafond de 100 s a disparu avec RunPod

Aucun appel `.stream()` dans la couche application. Toutes les complétions
sont bloquantes, timeout client 600 s.

**Le plafond n'existe plus.** L'inférence passe désormais par un Modal Server,
dont la documentation est explicite : il n'y a pas de timeout de plateforme,
c'est le client qui fixe le sien. Les 100 s de Cloudflare et les 524
appartenaient au proxy HTTP de RunPod. Le point reste listé pour une seule
raison : le code de streaming existe dans l'adaptateur, documente un problème
qui n'est plus le nôtre, et n'est toujours branché nulle part. Ce n'est plus un
piège, c'est du code mort à décider.

Ce que Modal impose à la place, et qui est nouveau : **un pool vide répond
503 immédiatement**, il ne met pas en attente. C'est traité
(`INFERENCE_SCALE_TO_ZERO`), vérifié en conditions réelles, et documenté dans
`gpu-worker/docs/modal.md`.

---

## 4. L'estimation de tokens est grossière, et tout en dépend

```
CHARS_PER_TOKEN = 4
```

Le code le dit lui-même : l'estimation peut se tromper d'un tiers dans les
deux sens. Or elle décide de deux choses :

- le budget de contexte (combien de fichiers entrent dans le prompt) ;
- la vérification de place du scheduler (`fits`).

**Ce n'est plus une hypothèse.** Le premier vrai run contre le GPU est mort
là-dessus, à un token près :

```
maximum context length is 32768 tokens. However, you requested 4096 output
tokens and your prompt contains at least 28673 input tokens
```

28672 est exactement 32768 − 4096, c'est-à-dire ce que `prompt_budget` renvoie :
de la place pour **tout** le prompt. L'orchestrateur la donnait telle quelle au
fournisseur de contexte comme budget de l'extrait de code, donc l'extrait
remplissait la fenêtre entière et le gabarit — instructions, objectif, plan,
findings accumulés, schéma JSON — la faisait déborder. Rien de tout cela
n'était compté.

Corrigé : `prompt_overhead_tokens` (2048) sort du budget de la flotte avant
qu'il devienne un budget d'extrait, et l'arithmétique vit dans
`OrchestratorConfig.excerpt_budget` pour qu'un test l'appelle au lieu de la
réécrire. Ce qui **reste** vrai, c'est la cause profonde : l'estimation divise
toujours des caractères par quatre. La vraie correction serait de compter avec
le tokenizer du modèle, ce qui obligerait cette couche à connaître le modèle.
Non fait, délibérément.

---

## 5. Le bac à sable des outils ne sait construire que du Python

```
tool_sandbox_image = "python:3.12-slim"
```

Une image **par déploiement**, pas par projet. Conséquences vérifiées :

- un projet C avec Makefile échoue en `make: not found` ;
- un projet Python échoue en `pytest: not found`, parce que l'image n'a que
  l'interpréteur.

`run-on.sh` interroge désormais le conteneur et **refuse le run avant de le
dépenser** en nommant ce qui manque, au lieu de découvrir le problème à la
validation. C'est un garde-fou, pas une solution.

La vraie correction est de porter l'image sur `ToolchainConfig`, ce qui est un
changement de schéma (table `projects`, schéma HTTP, migration). Non fait.

---

## 6. La toolchain d'un projet est corrigeable — ~~figée à sa création~~

Créer un projet qui existe déjà renvoie toujours l'enregistrement existant,
**inchangé** : la création n'est pas une mise à jour. Un run avait été dépensé
à exécuter `pytest` après que l'appelant eut remplacé cette commande.

`PUT /v1/projects/{id}/toolchain` remplace désormais les commandes, et elles
seules : le chemin, la branche et le nom identifient le code sur lequel
l'historique des runs a été produit, ils ne sont pas dans la charge utile
(`422` si on les y glisse). Le remplacement est total — une commande omise est
une commande supprimée. Refusé (`409 project_not_modifiable`) tant qu'un run du
projet est en vol : un run relit ces commandes à chaque validation de candidat,
et les changer sous lui ferait juger ses candidats selon deux définitions de
« ça passe ». `run-on.sh` corrige maintenant au lieu de refuser, et ne propose
`NAME=` que si l'API refuse.

---

## 7. Les artefacts déjà suivis par git restent dans le patch

Les exclusions de sortie de build s'appliquent à l'indexation et au diff. Un
fichier **déjà suivi** — un `.pyc` commité avant que la règle n'existe —
continue d'apparaître. C'est le comportement correct : le désindexer serait une
modification que l'agent n'a pas demandée. À signaler parce que c'est
déroutant quand on l'observe.

---

## 8. Code mort : la révision de plan

`PLAN_READY → PLANNING` est une transition légale de la machine à états, et
**rien ne la déclenche**. En conséquence `revision_reason` n'est jamais passé
au planner, alors que `previous_plan` l'est.

Ce n'est pas un bug aujourd'hui, c'est une moitié de fonctionnalité. Soit on la
branche (le planner saurait pourquoi on lui redemande un plan), soit on la
retire. La laisser telle quelle, c'est du code qui a l'air de marcher.

---

## 9. Lectures de dépôt jamais utilisées — supprimées

Les quatre méthodes que la version précédente de cette section listait **sont
retirées** : de leur déclaration dans `domain/ports/repositories.py`, de
l'adaptateur SQLAlchemy et du double en mémoire des tests. Un port déclaré que
personne n'appelle est une promesse que le code ne tient pas, et chaque
implémentation la paie.

| méthode retirée | pourquoi elle ne manquait à personne |
|---|---|
| `runs.count_by_status` | **vérifié** : `git log -S` la montre déclarée au tout premier commit de ports et jamais retouchée depuis ; aucune route n'agrège quoi que ce soit (`/health`, `/ready`, projets, runs, candidats, revues, workers, SSE), et la jauge `active_runs` de `telemetry/metrics.py` n'est alimentée par personne — aucun `.gauge(` dans `src/`. Le tableau de bord n'était pas en cours d'écriture : il n'a jamais été commencé. |
| `reviews.latest_for_candidate` | **vérifié** : les trois lecteurs de revues (réparation du coder, contexte du reviewer, `ListReviewsUseCase`) veulent l'historique entier et appellent `list_by_candidate`, qui trie par itération. La dernière revue, c'est son dernier élément. |
| `jobs.list_by_status` | **supposé** : aucun appelant, et aucun besoin visible — la file Redis est la source de vérité des jobs en attente ; une liste PostgreSQL par statut serait un outil de réconciliation, qui n'existe pas. |
| `jobs.list_expired_leases` | **vérifié** : la récupération des baux passe entièrement par `MaintenanceLoop.tick`, qui interroge `queue.reclaim_expired`. Son docstring la disait « destinée au redémarrage » ; or le redémarrage ne l'appelait pas (`resume_active_runs` ne lit que `list_active`), et la file survit au redémarrage — `redis-server --appendonly yes` dans le compose. |

Ce qui reste flaggé par le script, et le restera :

| méthode gardée | pourquoi |
|---|---|
| `UnitOfWork.rollback` | Faux positif. Son seul appelant est `__aexit__`, qui vit dans l'adaptateur, pas dans un cas d'usage — le script ne regarde que `application/` et `interfaces/`. Une frontière transactionnelle qui ne sait que valider n'en est pas une : le rollback implicite sur exception *est* cette méthode. La raison est écrite dans son docstring, là où le prochain lecteur la trouvera. |

Aucune de ces lectures n'était de la gravité de `tool_results.list_by_candidate`,
qui était la preuve déterministe écrite en base et relue par personne. Mais la
même mécanique les a trouvées, et elle vaut la peine d'être relancée après
chaque ajout de port :

```bash
./scripts/unused-ports.sh
```

**Vérifié** après la suppression : le script passe de « 22 port methods, 5 never
called » à « 18 port methods, 1 never called », les 806 tests de la suite
passent, `mypy` ne trouve rien et `ruff` est propre sur les fichiers touchés.

Les tests qui couvraient les méthodes retirées étaient tous dans
`tests/infrastructure/test_database_repositories.py`, et vérifiaient
l'adaptateur SQLAlchemy, pas un comportement métier :
`test_list_active_excludes_terminal_runs_and_counts_by_status` (renommé, sa
moitié `list_active` reste), la ligne `list_by_status` de
`test_job_round_trip_and_listings`, `test_reviews_are_listed_in_iteration_order`
(les assertions sur la dernière revue passent maintenant par `listed[-1]`, donc
la couverture du mapper est conservée) et
`test_list_expired_leases_returns_only_stuck_in_flight_jobs`, supprimé : il
n'existait que pour exercer la méthode retirée.

---

## 10. Interface : ce qui manque

L'interface (`web/`, Vite + React) couvre le suivi en direct, l'arborescence
superposée, la comparaison des candidats et l'approbation du merge.

Ce qu'elle n'a pas :

- **aucune liste globale des runs** — seulement par projet. Il faut choisir un
  projet avant de voir quoi que ce soit ;
- **aucune pagination** : `limit=50` en dur sur les runs ;
- **la flotte est interrogée toutes les 5 s** plutôt que suivie par événements,
  alors que les événements worker existent. Choix assumé (la flotte doit être
  juste *avant* d'ouvrir un run, pas seulement pendant), mais c'est du
  polling ;
- **aucune reprise du flux** si l'onglet reste fermé longtemps : `EventSource`
  reprend depuis `Last-Event-ID`, ce qui marche, mais rien n'affiche un
  avertissement si le rejeu est tronqué par `max_events_backfill` (500).

---

## 11. Sécurité : ce qui est ouvert par défaut

- `CORS_ALLOW_ORIGINS` est **vide** par défaut, donc aucun navigateur ne peut
  appeler l'API. C'est volontaire. Il faut l'ouvrir explicitement pour
  l'interface.
- `INFERENCE_API_KEY` vide signifie que le control plane n'envoie aucun bearer.
  Si le pod est lancé sans `VLLM_API_KEY`, **le port d'inférence est ouvert à
  qui peut l'atteindre** — sur un pod RunPod avec un port exposé, c'est
  l'internet. La documentation de l'image le dit ; rien ne l'empêche.
- `SERVICE_TOKEN` : la configuration **refuse un token vide** en production, et
  refuse aussi le provider factice (`config.py:173-176`). Ce qu'elle ne refuse
  pas, c'est la valeur par défaut du fichier compose,
  `dev-service-token-change-me` : elle n'est pas vide, donc elle passe. Un
  déploiement qui oublie de la changer est authentifié par un secret publié
  dans le dépôt.

---

## 12. Trois tentatives brûlées en quatre secondes

Observé le 22 septembre au soir, sur un run réel contre la H100 Modal. Le
worker a été déclaré indisponible par le reaper le temps qu'il se
ré-enregistre, et le job coder a consommé ses trois tentatives à 21:20:24,
:26 et :28 — **quatre secondes**, sans le moindre délai entre elles.

```
job ... (CODE) failed: no compatible worker is available -> RETRY_OTHER_WORKER
job ... (CODE) failed: no compatible worker is available -> RETRY_OTHER_WORKER
job ... (CODE) failed: no compatible worker is available -> FAIL
```

`RETRY_OTHER_WORKER` suppose qu'il existe un autre worker. Avec une flotte d'un
seul GPU — ce que le scale-to-zero rend normal, pas exceptionnel — il n'y en a
pas, et réessayer immédiatement trois fois revient à échouer une fois avec plus
d'étapes. Le run avait déjà fait planifier, coder, construire et tester **deux
candidats complets** ; il est mort sur une absence qui a duré moins d'une
minute.

**Corrigé à moitié, et la moitié qui reste est chiffrée.**

Le délai est désormais réel. `RetryPolicy` en calculait un depuis toujours et
`release()` le jetait : la file accepte maintenant un `not_before`, le job
attend dans un ensemble séparé par type, et il rejoint sa place **avec son
score d'origine** — attendre est une pause, pas une rétrogradation. Un job en
attente reste compté dans `depth()`, parce que c'est le chiffre qu'on lit pour
savoir si quelque chose est coincé. La promotion se fait dans le script de
réclamation lui-même, pas dans un balayeur, donc il n'existe aucun instant où
un job dû n'appartient à aucun ensemble. Les deux adaptateurs sont tenus par la
même suite de contrat, et elle a d'ailleurs attrapé un désaccord entre eux
avant que je le voie.

**La grandeur du délai aussi.** La politique produisait 1 s puis 2 s sur un
échec d'infrastructure, et trois tentatives espacées ainsi ne survivent pas à un
démarrage à froid de **130 s** — chiffre mesuré. « Aucun worker compatible » est
désormais un `FailureKind` à part entière : c'est le seul échec pour lequel la
seule chose qui puisse changer est le temps. Un bail expiré ou une connexion
refusée peuvent très bien réussir immédiatement ailleurs ; un pool vide répond
la même chose quelle que soit la vitesse à laquelle on l'interroge.

L'attente est **plate** (45 s) et non exponentielle, parce que ce qu'on attend a
une durée à peu près fixe — un démarrage à froid — et non inconnue. Elle est
calée sous le délai de heartbeat : un worker qui se lève s'annonce dans cette
fenêtre, donc attendre davantage n'achète qu'un job oisif. Le budget reste
borné : une flotte qui ne revient jamais fait échouer le run au lieu de le
tenir ouvert indéfiniment.

**Vérifié sur la pile réelle, Redis compris**, en remettant la flotte à zéro
et en créant un run — ce qui ne coûte pas un centime de GPU, puisque aucun
modèle n'est appelé :

```
avant            après
11:20:24         11:13:47   RETRY_OTHER_WORKER
11:20:26         11:14:33   RETRY_OTHER_WORKER      (+46 s)
11:20:28         11:15:19   FAIL                    (+46 s)
```

Et l'exécution a trouvé un second bug que les tests ne voyaient pas. Deux lignes
après le `release(not_before=…)`, l'orchestrateur republie le job avec
`enqueue()`, ce qui le remettait aussitôt dans l'ensemble « prêt » et annulait
le délai qui venait d'être posé. Un requeue est **deux appels**, et tester
`release` isolément restait vert pendant que la production ne l'était pas. Le
double de test ne surveillait que `release` : il était aveugle précisément au
bug qui comptait. La suite de contrat rejoue maintenant la séquence complète.

---

## 13. Le format « fichiers entiers » ne survit pas à un gros fichier

C'est la découverte la plus importante des runs du 22 septembre au soir, et
seule une exécution réelle pouvait la produire.

Objectif donné au coder : ajouter un validateur de dix lignes dans
`src/bootstrap/config.py` (357 lignes), plus deux tests. Les **quatre premières
lignes du patch sont exactement ce qui était demandé** — la condition, le
message, le nom de la variable :

```python
if self.environment.is_production and self.service_token.get_secret_value() == "dev-service-token-change-me":
    raise ValueError(
        "SERVICE_TOKEN is set to the published default value. "
        "This value is public and anyone can authenticate with it. "
```

Puis, dans le même fichier, il a **réécrit entièrement la classe
`WorkerSettings`** à laquelle on ne lui demandait rien. Il a perdu des champs
qui existaient (`model_id`, `model_context_length`, `llm_provider`,
`worker_concurrency`, `gpu_type`), il y a recopié des champs qui appartiennent à
`Settings` (`workspace_root`, `artifact_root`, `executor_concurrency`,
`reaper_interval_seconds`, `static_analysis_is_blocking`), et il a ajouté un
`_check_coherence` qui référence `heartbeat_interval_seconds` sans le déclarer.
Résultat :

```
PydanticUserError: check_decorator_fields_exist
```

Le module ne s'importe plus, donc **tous** les tests échouent à la collecte, et
les trois tours de réparation n'y ont rien changé : chaque tour recevait la même
erreur d'import et réécrivait le même fichier entier. 224 lignes de churn pour
une modification qui en demandait une quinzaine.

Ce que cela dit du format : réécrire un fichier entier oblige le modèle à
reproduire de mémoire tout ce qu'il ne touche pas. Sur 50 lignes c'est gratuit,
et le rapport le notait comme un succès. Sur 357 lignes contenant deux classes
de configuration qui se ressemblent, il reconstruit la mauvaise.

**Pistes, aucune faite :** passer à un format d'édition localisée pour les
fichiers au-delà d'un certain seuil ; ou refuser en validation un patch qui
touche des symboles absents des `target_paths` du planner ; ou, au minimum,
compter cela comme un signal distinct de « les tests échouent » — un module qui
ne s'importe plus n'est pas un test rouge, c'est un fichier cassé, et le coder
gagnerait à ce qu'on le lui dise dans ces termes.

## Ce qui a été vérifié et fonctionne

Pour l'équilibre, et parce que ces points ne doivent pas être re-testés à
chaque doute :

- **la réconciliation modèle/fenêtre** : l'agent lit `/v1/models` et annonce ce
  que vLLM sert réellement. Vu sur la machine, avec un pod configuré exprès
  pour contredire l'agent :
  `declared 131072, but the engine serves 32768` et
  `declared 'Qwen/...-Instruct', but the engine serves 'qwen3-coder'` ;
- **la sortie structurée** via `response_format: json_schema` : planner, coder
  et reviewer ont tous produit du JSON valide contre vLLM 0.28 ;
- **le format « fichiers entiers »** : patch propre et diffable, 50 lignes,
  zéro artefact sur un dépôt propre — **mais voir le point 13, qui montre où
  ce format cesse de tenir** ;
- **la clé d'inférence** arrive jusqu'à l'adaptateur (ce ne fut pas toujours
  le cas : elle était câblée dans le code et absente du compose) ;
- **753 tests**, ruff et mypy propres.

---

## Priorités suggérées

1. ~~Relancer un run réel~~ **Fait le 22 septembre au soir**, trois fois, contre
   une H100 Modal. La chaîne complète fonctionne : plan, code, build, test,
   réparation. Le point 13 est ce qu'il faut corriger en premier — c'est lui qui
   a fait échouer le run le plus abouti, et il touche tout objectif qui vise un
   fichier de plus de quelques centaines de lignes.
2. **Mesurer** le nombre de tours de réparation avec la nouvelle information
   avant de toucher `max_repair_iterations`.
3. **Image de bac à sable par projet** (point 5) si l'on veut autre chose que
   du Python. C'est le seul point qui bloque une famille entière de projets.
4. Le reste est du confort ou du durcissement.
