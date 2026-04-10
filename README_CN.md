# ZiSA 项目中文使用说明

这个文档是给第一次接触 Git 和 GitHub 的用户准备的。你可以把它理解成这个项目的“入门操作手册”。

如果你只想记住最重要的内容，只看这一句：

```powershell
git add .
git commit -m "写本次修改说明"
git push
```

但在真正使用前，建议先把下面内容看一遍。

## 1. 这个项目在哪里操作

你要操作的项目目录是：

```powershell
D:\Pythoncode\VMamba-main\VMamba-main
```

注意：

- 外层目录 `D:\Pythoncode\VMamba-main` 不是 Git 仓库
- 内层目录 `D:\Pythoncode\VMamba-main\VMamba-main` 才是 Git 仓库

所以以后打开终端后，第一步通常都是：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
```

## 2. Git 和 GitHub 是什么

可以简单理解成：

- `Git`：本地版本管理工具，负责记录你每次改了什么
- `GitHub`：远程网站仓库，负责把你的代码保存到云端

你在电脑上改完代码后，要做两件事：

1. 先用 Git 在本地保存一次修改记录
2. 再推送到 GitHub

## 3. 你以后最常用的 4 条命令

每次你改完项目后，最常用的是这几条：

```powershell
git status
git add .
git commit -m "这里写你这次修改的说明"
git push
```

它们分别是什么意思：

- `git status`：查看你改了哪些文件
- `git add .`：把当前修改加入“待提交列表”
- `git commit -m "说明"`：正式保存一次本地修改记录
- `git push`：把本地提交上传到 GitHub

## 4. 最适合新手的完整更新流程

假设你已经改好了代码，比如改了 `README.md`、训练脚本或者配置文件，那么请按下面顺序来：

### 第一步：进入项目目录

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
```

### 第二步：查看当前修改

```powershell
git status
```

你会看到类似这样的信息：

```powershell
modified:   README.md
modified:   detection/train_windfarm_detection.py
```

这表示这些文件被你改过，但还没有保存到 Git 记录里。

### 第三步：把修改加入暂存区

如果你想把当前所有修改一起提交：

```powershell
git add .
```

如果你只想提交某一个文件：

```powershell
git add README.md
```

或者：

```powershell
git add detection/train_windfarm_detection.py
```

### 第四步：提交到本地 Git

```powershell
git commit -m "更新README和训练脚本"
```

这一步很重要。`commit` 就像“存档”。

你每做一次 `commit`，Git 就会记住你这次改动。

### 第五步：上传到 GitHub

```powershell
git push
```

这一步执行成功后，你打开 GitHub 仓库页面，就能看到更新后的内容。

## 5. 推荐你直接照着用的日常模板

以后你每次改完代码，直接复制这一套：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git status
git add .
git commit -m "写本次更新内容"
git push
```

例如：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git status
git add .
git commit -m "修改ZiSA模型配置并更新README"
git push
```

## 6. 如果你只改了一个文件

比如你只改了 `README.md`，那就可以这样：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git status
git add README.md
git commit -m "更新项目说明文档"
git push
```

这样会更干净，不会把其他没准备好的文件一起提交。

## 7. 如何看自己现在有没有改东西

执行：

```powershell
git status
```

常见情况有两种。

### 情况 1：有修改

如果看到：

```powershell
modified: README.md
```

说明你有文件改过，还没有提交。

### 情况 2：没有修改

如果看到：

```powershell
nothing to commit, working tree clean
```

说明当前没有新的改动，不需要提交。

## 8. 提交说明怎么写

这一句：

```powershell
git commit -m "这里写说明"
```

里面的说明，建议写清楚你改了什么。不要只写“更新”两个字，最好写得具体一点。

例如：

```powershell
git commit -m "更新README说明"
git commit -m "修改风电检测训练配置"
git commit -m "新增ZiSA可视化脚本"
git commit -m "修复模型导入路径问题"
```

## 9. 如果推送失败怎么办

### 情况 1：网络问题

如果报错和网络有关，就重新执行一次：

```powershell
git push
```

### 情况 2：GitHub 登录或权限问题

如果提示需要认证、权限不足、账号未登录，一般需要：

1. 确认你登录的是自己的 GitHub 账号
2. 确认这个仓库是你自己的仓库，或者你有权限
3. 再重新执行 `git push`

### 情况 3：远程有新内容，本地没同步

可以先执行：

```powershell
git pull --rebase
```

然后再执行：

```powershell
git push
```

## 10. 推荐新手使用的安全流程

为了尽量减少出错，推荐你每次都按这个流程来：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git status
git pull --rebase
git status
git add .
git commit -m "写本次修改说明"
git push
```

说明：

- `git pull --rebase`：先把 GitHub 上最新内容同步到本地
- 然后再提交自己的修改
- 最后再推送

如果你是一直只在这一台电脑上改代码，这一步有时不是必须，但养成习惯更好。

## 11. 这个项目第一次克隆以后，平时怎么更新

如果仓库已经在你电脑里了，以后不需要再重复 `git init` 或重新绑定远程仓库。

你只需要做这几件事：

1. 打开终端
2. 进入项目目录
3. 修改代码
4. `git add`
5. `git commit`
6. `git push`

也就是说，以后你最常用的是：

```powershell
git status
git add .
git commit -m "说明"
git push
```

## 12. 千万别在错误目录执行命令

你之前遇到过这个问题，所以这里单独提醒。

错误目录：

```powershell
D:\Pythoncode\VMamba-main
```

正确目录：

```powershell
D:\Pythoncode\VMamba-main\VMamba-main
```

如果你不确定自己在哪个目录，可以执行：

```powershell
pwd
```

或者直接重新进入正确目录：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
```

## 13. 这个项目最常用的一套命令

你可以把下面这段当成固定模板：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git status
git add .
git commit -m "本次修改内容"
git push
```

## 14. 如果你现在就要提交这个中文文档

可以直接执行：

```powershell
cd D:\Pythoncode\VMamba-main\VMamba-main
git add README_CN.md
git commit -m "新增中文Git使用说明"
git push
```

## 15. 一句话总结

以后你每次改完项目，记住这 4 步：

1. `git add .`
2. `git commit -m "写清楚你改了什么"`
3. `git push`
4. 去 GitHub 页面刷新查看结果

如果你愿意，我下一步可以继续帮你把顶层 `README.md` 加一个“中文说明入口”，让别人打开仓库就能看到这个中文文档。
