# Data samples that gonna be used to calculate the similarity
samples = dict(
    recommendation = [
                      "Je recommande et j'y retournerai avec grand plaisir.",
                      'Nous y retournerons sans hésiter je recommande!',
                      'excellentes et délicieuses que nous conseillons vivement.',
                      'agréable, dans un cadre authentique, je recommande fortement',
                      'Je conseille vivement cet endroit pour passer un bon moment en toute sécurité et de convivialité.','Je le conseil fortement à tous.',
                      "Je recommande vivement !",
                      'je recommande et je reviendrais',
                      'Dans un coin de nature, je recommande vivement.',
                      'Je recommande chaleureusement.',
                      'on reviendra car nous sommes pas loin !',
                      'Nous recommandons vivement cet endroit pour passer un très bon séjour à deux.',
                      "nous n'hésiterons pas à revenir.",
                      'Délicieux, je recommande.',
                      "j’y vais souvent c’est très très bon et bien pour la santé je vous conseille fortement",
                      'je recommande fortement ce lieux pour une jolie parenthèse enchantée .',
                      'J’y retournerai et je recommande sincèrement.',
                      'Super resto que je recommande.',
                      'Bref, très bon moment, je recommande à 100% et y retournerai volontier.',
                      'Nous recommandons et reviendrons !',
                      "je recommande fortement ce restau c'est tellement bon",
                      'Nous le conseillons a 1000% et nous reviendrons assez vite le faire découvrir a nos familles et autres amis.',
                      'Bref tout est parfait je recommande vivement .',
                      'Nous recommandons cet établissement.'
                      ],
    problems = [
                     "Toilettes peu pratiques pour les grandes personnes : trop proches de la douche",
                      "La chambre est toute petite, ainsi que la salle de douche (Pour illustration: l'évacuation de la douche est situé sous les WC, on se douche donc sur les WC",
                      "Très bel hôtel moderne salle de douche avec une bonne pression dans la douche.",
                      "la salle de bain est minuscule pour le prix on pourrait s'attendre à un lit king size manque de prise électrique",
                      "la cabine de douche trop petite et peu de pression d'eau",
                      "La salle de bain est trop minuscule, pas de savon,le bac de douche trop petit.",
                      "Seul petit détail qui manque : une prise électrique à côté du lit",
                      "Pas d'ascenseur, pas de clim, et mini salle de bains",
                      "Le lit étai vraiment une horreur on c'est levé avec un mal de dos touts les deux.",
                      "Un peu de bruit dans le couloir au début de la nuit.",
                      "La fuite d'eau au-dessus du comptoir d'accueil.",
                      "Pas d'ascenseur pour un hôtel avec 2 étages",
                      "La température de l’eau très instable et le manque de pression de l’eau rendent la douche inconfortable",
                      "Réseau téléphone mobile très faible ou pas de réseau",
                      "Le manque de rideau de douche dans la salle d'eau.",
                      "Trop peu de pression d'eau dans la douche, avec température d'eau insuffisamment chaude",
                      "La salle de douche était minuscule et sale aussi.",
                      "Absence de prise de courant près du lit.",
                      "La propreté de la chambre, le manque de prise électrique notamment dans la salle de bain, le manque d'un balais de toilette dans la salle de bain, le déclenchement de l'alarme incendie à 7h30 du matin",
                      "La salle de bain beaucoup trop petite, pas de placard et penderie pour ranger les affaires",
                      "Il manque une chaise de plus la chambre est pour deux et aussi une prise électrique dans la salle de bains",
                      "Problème de pression d eau et eau chaude dans la salle de bain",
                      "Pas d’ascenseur pour les bagages",
                      "Pas wifi dans la chambre , pas de prise de courant disponible, sauf 1 salle de bain",
                      "Manque rideau de douche ou pare douche",
                      "Code Wifi : nb maximal de connexions atteint antérieurement.",
                      "Ils ne nous ont pas remboursé les petits déjeuner que nous n'avions pas pu consommer, disant que ça viendrait pus tard.... il y a 3 jours.",
                      "La chambre très petite pour 3 personnes, et la fenêtre donnant sur un mur, pas très agréable comme vue.",
                      "Un seul bémol, il manque de pression pour l’eau dans la douche, ainsi que de l’eau chaude si vous prenez votre bain le soir...",
                      "Pas de barre d’appui dans la baignoire glissante et flexible de douche trop court",
                      "Manquecomment_idx de prises de courant dans la chambre et d'étagères dans le salle de bain.",
                      "Il manque un rideau pour la baignoire et une barre d’appui pour ne pas glisser",
                      "Nous n'avons pas eu le droit a notre petit déjeuner, alors que nous avons payer la formule pour pouvoir déjeuner.."
                      ]
               )
# Settings of the solution
settings = dict(
    bert_model = 'paraphrase-xlm-r-multilingual-v1',
    sample_embedds_path = "./sample_embedds/",
    threshold = 0.77
)