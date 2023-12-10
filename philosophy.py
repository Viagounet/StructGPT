from engine import Engine
import yaml

with open("examples/parameters/philosophy.yaml", "r") as file:
    parameters = yaml.safe_load(file)


text = """Mais ici surgit la tâche difficile de faire comprendre comment une aperception pareille est possible et pourquoi n’est-elle pas plutôt tout de suite annulée? 
Comment se fait-il que, conformément aux faits, le sens transféré est accepté comme ayant une valeur existentielle, comme ensemble de déterminations « psychiques » du corps de l’ autre, 
tandis que celles-ci ne peuvent jamais se montrer, en elles-mêmes, dans le domaine original de ma sphère primordiale (le seul qui est à notre disposition). 
Considérons de plus près la situation intentionnelle. L ’apprésentation qui nous donne ce qui, en autrui, nous est inaccessible en original, est liée à une présentation originelle 
(de son corps, élément constitutif de ma nature, donnée comme « m’appartenant »)."""

template = """- <Partie 1>
- <Partie 2>
- <Partie 3>"""
engine = Engine("gpt-4", parameters=parameters)
# ans = engine.query(
#     f"Texte:\n{text}\n---\nTemplate:\n{template}\n---\nEcrire le plan d'explication du texte en suivant la template.",
#     max_tokens=400,
# )
# print(ans)
# print("--------------")
# for i in range(3):
#     ans = engine.query(
#         f"Texte:\n{text}\n---\nPlan général:{ans}\nPour chaque partie du plan, détailler en sous-parties (indenter les parties détaillées).",
#         max_tokens=800 * (i + 1),
#     )
#     print(ans)
#     print("--------------")
# ans = engine.query(
#     f"Texte:\n{text}\n---\nPlan général:{ans}\nExpliquer le texte original en suivant scrupuleusement les différentes parties et sous-parties du plan. Utiliser des mots de liaison pour relier chaque partie entre elles.",
#     max_tokens=4096,
# )
# print("==========")
# print(ans)

plan = """Partie 1: Analyse de la question de l'appréhension de l'autre et de la difficulté de comprendre comment une telle perception est possible. Discussion sur pourquoi cette perception n'est pas immédiatement annulée.
    - Sous-partie 1.1: Définition et explication de la notion d'appréhension de l'autre.
        - Point 1.1.1: Définition de l'appréhension de l'autre.
            - Sous-point 1.1.1.1: Explication du concept d'appréhension.
            - Sous-point 1.1.1.2: Discussion sur la spécificité de l'appréhension de l'autre.
        - Point 1.1.2: Explication de l'importance de l'appréhension de l'autre dans la perception de soi et du monde.
            - Sous-point 1.1.2.1: Analyse de l'impact de l'appréhension de l'autre sur la perception de soi.
            - Sous-point 1.1.2.2: Discussion sur le rôle de l'appréhension de l'autre dans la perception du monde.
    - Sous-partie 1.2: Discussion sur la complexité de comprendre comment une telle perception est possible.
        - Point 1.2.1: Analyse des facteurs qui rendent cette perception possible.
            - Sous-point 1.2.1.1: Exploration des facteurs cognitifs.
            - Sous-point 1.2.1.2: Analyse des facteurs sociaux et culturels.
        - Point 1.2.2: Discussion sur les défis inhérents à la compréhension de cette perception.
            - Sous-point 1.2.2.1: Identification des obstacles à la compréhension.
            - Sous-point 1.2.2.2: Discussion sur les possibles solutions pour surmonter ces obstacles.
    - Sous-partie 1.3: Analyse des raisons pour lesquelles cette perception n'est pas immédiatement annulée.
        - Point 1.3.1: Exploration des raisons pour lesquelles cette perception persiste.
            - Sous-point 1.3.1.1: Analyse des bénéfices de la persistance de cette perception.
            - Sous-point 1.3.1.2: Discussion sur les mécanismes qui soutiennent cette persistance.
        - Point 1.3.2: Analyse des conséquences de l'annulation de cette perception.
            - Sous-point 1.3.2.1: Exploration des impacts négatifs potentiels de l'annulation.
            - Sous-point 1.3.2.2: Discussion sur les conséquences positives possibles de l'annulation.

Partie 2: Exploration de la manière dont le sens transféré est accepté comme ayant une valeur existentielle, en tant qu'ensemble de déterminations "psychiques" du corps de l'autre, malgré le fait qu'elles ne peuvent jamais se montrer dans le domaine original de notre sphère primordiale.
    - Sous-partie 2.1: Explication de la notion de sens transféré et de sa valeur existentielle.
        - Point 2.1.1: Définition du sens transféré.
            - Sous-point 2.1.1.1: Explication du concept de sens transféré.
            - Sous-point 2.1.1.2: Discussion sur les différentes formes de sens transféré.
        - Point 2.1.2: Discussion sur la valeur existentielle du sens transféré.
            - Sous-point 2.1.2.1: Analyse de l'importance du sens transféré dans l'existence humaine.
            - Sous-point 2.1.2.2: Discussion sur les implications de la valeur existentielle du sens transféré.
    - Sous-partie 2.2: Discussion sur le rôle des déterminations "psychiques" du corps de l'autre.
        - Point 2.2.1: Explication des déterminations "psychiques" du corps de l'autre.
            - Sous-point 2.2.1.1: Définition des déterminations "psychiques".
            - Sous-point 2.2.1.2: Discussion sur la spécificité des déterminations "psychiques" du corps de l'autre.
        - Point 2.2.2: Analyse du rôle de ces déterminations dans la perception de l'autre.
            - Sous-point 2.2.2.1: Exploration de l'impact de ces déterminations sur la perception de l'autre.
            - Sous-point 2.2.2.2: Discussion sur les mécanismes par lesquels ces déterminations influencent la perception.
    - Sous-partie 2.3: Analyse de la contradiction apparente entre l'acceptation de ces déterminations et leur absence dans le domaine original de notre sphère primordiale.
        - Point 2.3.1: Présentation de la contradiction apparente.
            - Sous-point 2.3.1.1: Description de la contradiction.
            - Sous-point 2.3.1.2: Discussion sur les implications de cette contradiction.
        - Point 2.3.2: Analyse des raisons possibles de cette contradiction.
            - Sous-point 2.3.2.1: Exploration des facteurs qui pourraient expliquer cette contradiction.
            - Sous-point 2.3.2.2: Discussion sur les possibles résolutions de cette contradiction.

Partie 3: Examen plus approfondi de la situation intentionnelle, en mettant l'accent sur l'apprésentation qui nous donne ce qui est inaccessible en autrui, en lien avec une présentation originelle de son corps.
    - Sous-partie 3.1: Définition et explication de la situation intentionnelle.
        - Point 3.1.1: Définition de la situation intentionnelle.
            - Sous-point 3.1.1.1: Explication du concept de situation intentionnelle.
            - Sous-point 3.1.1.2: Discussion sur les différentes formes de situations intentionnelles.
        - Point 3.1.2: Explication de l'importance de la situation intentionnelle dans la perception de l'autre.
            - Sous-point 3.1.2.1: Analyse de l'impact de la situation intentionnelle sur la perception de l'autre.
            - Sous-point 3.1.2.2: Discussion sur les mécanismes par lesquels la situation intentionnelle influence la perception.
    - Sous-partie 3.2: Analyse de l'apprésentation comme moyen d'accéder à ce qui est inaccessible en autrui.
        - Point 3.2.1: Définition de l'apprésentation.
            - Sous-point 3.2.1.1: Explication du concept d'apprésentation.
            - Sous-point 3.2.1.2: Discussion sur les différentes formes d'apprésentation.
        - Point 3.2.2: Discussion sur le rôle de l'apprésentation dans l'accès à l'inaccessible.
            - Sous-point 3.2.2.1: Analyse de l'efficacité de l'apprésentation pour accéder à l'inaccessible.
            - Sous-point 3.2.2.2: Discussion sur les limites de l'apprésentation.
    - Sous-partie 3.3: Discussion sur le lien entre cette appréhension et la présentation originelle du corps de l'autre.
        - Point 3.3.1: Explication de la présentation originelle du corps de l'autre.
            - Sous-point 3.3.1.1: Définition de la présentation originelle.
            - Sous-point 3.3.1.2: Discussion sur la spécificité de la présentation originelle du corps de l'autre.
        - Point 3.3.2: Analyse du lien entre l'appréhension de l'autre et cette présentation originelle.
            - Sous-point 3.3.2.1: Exploration de la relation entre l'appréhension de l'autre et la présentation originelle.
            - Sous-point 3.3.2.2: Discussion sur les implications de ce lien pour la perception de l'autre."""
ans = engine.query(
    f"Texte: {text}\n---\nPRédiger un commentaire de texte complet en précisant la pensée de l'auteur",
    max_tokens=5000,
)
print(ans)
