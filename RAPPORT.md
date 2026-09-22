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

## 3. Pas de streaming : plafond de 100 s sur le proxy HTTP RunPod

Aucun appel `.stream()` dans la couche application. Toutes les complétions
sont bloquantes, timeout client 600 s.

- **En tunnel TCP direct** — ce qui a été utilisé et ce que fait
  `scripts/retry-gpu-test.sh` (`--expose tcp`) — **sans objet**.
- **Via le proxy HTTP RunPod**, Cloudflare coupe à 100 s de *lecture* : toute
  génération plus longue devient un **524**.

Le code de streaming existe dans l'adaptateur et documente correctement le
problème. Il n'est simplement pas branché. Tant que le déploiement reste en
TCP, ce n'est pas urgent ; c'est un piège pour quiconque passera en proxy.

---

## 4. L'estimation de tokens est grossière, et tout en dépend

```
CHARS_PER_TOKEN = 4
```

Le code le dit lui-même : l'estimation peut se tromper d'un tiers dans les
deux sens. Or elle décide de deux choses :

- le budget de contexte (combien de fichiers entrent dans le prompt) ;
- la vérification de place du scheduler (`fits`).

Avec une fenêtre de 32768 et 4096 réservés pour la réponse, une
sous-estimation d'un tiers déborde. Le garde-fou existe — une réponse tronquée
échoue immédiatement avec un message qui nomme le budget, au lieu de brûler
trois générations — donc **on le verra**, mais on le verra.

La vraie correction serait de compter les tokens avec le tokenizer du modèle,
ce qui obligerait cette couche à connaître le modèle. Non fait, délibérément.

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

## 6. La toolchain d'un projet est figée à sa création

Il n'existe aucune route de mise à jour d'un projet. Créer un projet qui existe
déjà renvoie l'enregistrement existant, **inchangé**. Un run a été dépensé à
exécuter `pytest` après que l'appelant eut remplacé cette commande.

`run-on.sh` compare désormais et refuse en proposant `NAME=`. L'API, elle, ne
permet toujours pas de corriger un projet : il faut en créer un autre.

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

## 9. Lectures de dépôt jamais utilisées

Ports déclarés, jamais appelés depuis `application/` ni `interfaces/` :

| méthode | commentaire |
|---|---|
| `runs.count_by_status` | destinée à un tableau de bord qui n'existe pas |
| `reviews.latest_for_candidate` | supplantée par `list_by_candidate` |
| `jobs.list_by_status` | jamais nécessaire |
| `jobs.list_expired_leases` | la file fait sa propre récupération |
| `UnitOfWork.rollback` | appelée par le gestionnaire de contexte, pas par du code métier — faux positif du script |

Aucune n'est de la gravité de `tool_results.list_by_candidate`, qui était la
preuve déterministe écrite en base et relue par personne. Mais la même
mécanique les a trouvées, et elle vaut la peine d'être relancée après chaque
ajout de port :

```bash
./scripts/unused-ports.sh
```

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
  zéro artefact sur un dépôt propre ;
- **la clé d'inférence** arrive jusqu'à l'adaptateur (ce ne fut pas toujours
  le cas : elle était câblée dans le code et absente du compose) ;
- **753 tests**, ruff et mypy propres.

---

## Priorités suggérées

1. **Relancer un run réel** dès qu'il y a de la capacité GPU. Tout le reste de
   cette liste est de l'hypothèse tant que ce n'est pas fait.
2. **Mesurer** le nombre de tours de réparation avec la nouvelle information
   avant de toucher `max_repair_iterations`.
3. **Image de bac à sable par projet** (point 5) si l'on veut autre chose que
   du Python. C'est le seul point qui bloque une famille entière de projets.
4. Le reste est du confort ou du durcissement.
