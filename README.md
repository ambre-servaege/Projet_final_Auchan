# Projet Auchan - Assistant de Substitution IA

## Description
Application de thèse utilisant un modèle SVM et la similarité cosinus pour suggérer des alternatives aux produits Auchan.

## URL Publique
https://projetfinalauchan.streamlit.app/

## URL du dépôt Git
https://github.com/ambre-servaege/Projet_final_Auchan

## Prérequis
- Python 3.11+
- PostgreSQL
- Bibliothèques : `pip install -r requirements.txt`

## Installation
1. Cloner le dépôt :
   `git clone [VOTRE_URL]`
2. Installer les dépendances :
   `pip install -r requirements.txt`
3. Importer le dump SQL situé dans `/sql/export_db.sql` dans votre base PostgreSQL.
4. Configurer le fichier `/config/.env` avec vos identifiants de base de données.

## Utilisation
- Lancer l'application :
  `streamlit run src/app.py`
- Accès administrateur : Utilisateur `admin@auchan.fr` / Mot de passe `[VOTRE_MDP]`

## Compatibilité
Ce projet a été testé avec succès sur les navigateurs Chrome, Firefox et Safari.
