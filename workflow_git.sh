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