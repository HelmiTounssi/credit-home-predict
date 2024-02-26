# Déploiement du modèle
Le modèle a été déployé en tant que service web à l'aide de Flask et Dash. Le tableau de bord résultant permet à l'utilisateur d'interagir avec le modèle en soumettant des fichiers CSV ou Excel pour prédiction. Il s'agit d'une simulation plutôt simple de la façon dont un employé de banque pourrait utiliser un système comme celui-ci.

Notez également que nous avons actuellement codé en dur la version du modèle à utiliser dans le déploiement. Pour rationaliser ce processus, nous devrions viser à stocker le modèle et les artefacts dans le cloud (comme un bucket S3 ou Cloud Storage) que nous pouvons ensuite récupérer directement. Cela nous permettrait d'entraîner automatiquement puis de déployer le meilleur modèle de manière continue.

# Déploiement local
Utilisation d'un serveur de développement
Pour démarrer le tableau de bord, exécutez la commande suivante dans le Terminal 

bash

python ../model/app.py
Cela lance un serveur sur votre machine locale sur le port 8090. Rendez-vous sur http://127.0.0.1:8090/ à l'aide d'un navigateur et commencez à interagir avec le modèle.

Si ce n'est pas déjà fait, installez gunicorn, que nous utiliserons comme serveur de production, en utilisant pip install gunicorn. Ensuite, exécutez la commande suivante dans ce même dossier pour démarrer l'application :

bash

gunicorn --bind=localhost:8090 model_deployment.app:server
L'application est alors accessible via la même URL qu'auparavant.

Déploiement Docker
À partir du dossier principal du projet, où se trouve le Dockerfile, exécutez :

bash

docker build -t home-credit .
Suivi de

bash

docker run -it --rm -p 8090:8090 home-credit:latest
Cela démarrera le serveur. Accédez à http://0.0.0.0:8090/ pour voir et tester l'application en direct !

<br><br>

Test initial simple
Pour tester le modèle, exécutez ce qui suit dans ce dossier :

bash

python test_main.py
La précision, le rappel, la récupération et d'autres métriques devraient être imprimés dans le terminal. Ils devraient être alignés avec ceux de test_app.py.