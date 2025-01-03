---

### Steps to Save This `README.md` 


# Step 1: Initialize Git
git init

# Step 2: Stage your script file
# Use this command when you've created or updated a file and want to save it to Git's staging area:
git add starting.py

# Step 3: Commit the changes
# Use this command after staging files to create a snapshot of your work with a descriptive message
git commit -m "Describe what you changed or added"

# Step 4: Link to your GitHub repository
git remote add origin https://github.com/bfmilan/starting.git

# Step 5: Push the script to GitHub
git branch -M main
git push -u origin main

# Clone a repository when you want to create a local copy of an existing GitHub repository
git clone https://github.com/bfmilan/starting.git

# When and How to Create a New Branch: 1. When adding a new feature or making significant changes (e.g., testing a new visualization), 2. To isolate your changes and avoid affecting the main branch. 

# Create new branch locally: 
git branch feature-new-visualization
# switch to the new branch 
git checkout feature-new-visualization
# Push the branch if you want to share your work or collaborate with others:
git push -u origin feature-new-visualization

Typical Git Commands with Explanations

Check the current branch: Use this command to see which branch you're working on:
git branch
View commit history: Use this command to see all past commits in your project:
git log
Remove a file from Git but keep it locally: Use this command when you want Git to stop tracking a file, but you don’t want to delete it locally:
git rm --cached <filename>
Undo the last commit without losing changes: Use this command if you made a mistake in your last commit:
git reset --soft HEAD~1


