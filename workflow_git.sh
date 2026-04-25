git branch
git status
git pull
git checkout -b feature-x
git add .
git commit -m "$1"
git push
git checkout main
git merge feature-x
git push 
git branch -d feature-x
git push origin --delete feature-x

# ./git_push.sh "update code"
# Workflow pour supprimer les fichiers sensibles de l’historique Git et forcer le push du nouvel historique
# 1. Supprimer les fichiers sensibles de tout l’historique
git filter-repo --path src/data_download_part2/credentials.json --invert-paths
git filter-repo --path src/data_download_part2/token.json --invert-paths

# 2. Reconnecter le dépôt distant (origin)
git remote add origin https://github.com/chahinez26/Crop-Classification-Projet.git

# 3. Forcer le push du nouvel historique
git push origin main --force