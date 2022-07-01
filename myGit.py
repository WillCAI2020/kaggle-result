import os
from git.repo import Repo
from git.repo.fun import is_git_dir

if __name__ == '__main__':
    datasetName="CrackLS315"
    dataName= "train"
    netName = "crackformer"

    train_img_dir = "/kaggle/working/kaggle-crackFormer/datasets/"+ datasetName +"/" + dataName +".txt"
    valid_img_dir = "/kaggle/working/kaggle-crackFormer/datasets/"+datasetName+"/valid/50a/"
    valid_lab_dir = "/kaggle/working/kaggle-crackFormer/datasets/"+datasetName+"/valid/50b/"
    valid_result_dir = "/kaggle/working/kaggle-crackFormer/datasets/"+datasetName+ "/valid/50res/"
    valid_log_dir = "/kaggle/working/kaggle-crackFormer/log/" + netName 
    best_model_dir = "/kaggle/working/kaggle-crackFormer/model/" + datasetName +"/"
    
    local_path = '/kaggle/working/kaggle-crackFormer'
    git_local_path = os.path.join(local_path, '.git')
    if is_git_dir(git_local_path):
        print('开始上传结果')
        os.chdir(local_path)
        print(os.getcwd())
        repo = Repo(local_path) # 已经存在git仓库
        # repo.git.remote('add', 'origin', 'https://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@github.com/WillCAI2020/kaggle-result.git')
        # repo.git.add(valid_result_dir)
        Repo.git.add('main.py')
        # repo.git.add(best_model_dir)
        Repo.git.commit('-m', '提交')
        Repo.git.push()
        print('上传结束')