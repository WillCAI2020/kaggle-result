import os
from git.repo import Repo
from git.repo.fun import is_git_dir


class myGit(object):
    """
    git仓库管理
    """

    def __init__(self, local_path, repo_url, branch='main'):
        self.local_path = local_path
        self.repo_url = repo_url
        self.repo = None
        self.initial(repo_url, branch)

    def initial(self, repo_url, branch):
        """
        初始化git仓库
        :param repo_url:
        :param branch:
        :return:
        """
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)

        git_local_path = os.path.join(self.local_path, '.git')
        if is_git_dir(git_local_path):
            self.repo = Repo(self.local_path) # 已经存在git仓库
    
    
    
    def push(self):
        """
        推送到远程仓库
        :return:
        """
        self.repo.git.push()