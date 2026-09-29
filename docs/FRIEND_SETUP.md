# 分享版安装说明（Windows）

## 一、配置 JEV 环境变量

在 PowerShell 中运行：

```powershell
[Environment]::SetEnvironmentVariable("JEV_API_KEY", "你的 JEV API Key", "User")
```

关闭并重新打开 Codex，让新的环境变量生效。也可以只在当前 PowerShell 会话中临时设置：

```powershell
$env:JEV_API_KEY = "你的 JEV API Key"
```

不要把真实的 API Key 写入配置文件，也不要发到群聊或代码仓库中。

## 二、解压并打开项目

将压缩包解压到一个固定目录，在 Codex 中打开解压后的项目文件夹，并按照提示信任项目 hooks。

项目 hook 配置会记录用户需求和工具执行证据。对于明确的只读、编辑、测试或构建操作，程序会先使用本地预筛；只有必要时才调用 JEV。普通工具输出只在本地记录。本分享配置不包含 Stop hook。

## 三、可选的连接配置

默认使用示例配置中的 JEV 服务地址、`jev-latest` 模型和 30 秒客户端超时。如需修改服务地址或模型，可以在项目根目录创建 `jev.config.json`：

```json
{
  "base_url": "https://api.typesafe.ai/v1/systemone",
  "model": "jev-latest",
  "timeout": 30
}
```

只需要 API Key 时不必创建该文件。必需的环境变量名是 `JEV_API_KEY`；只有在把配置文件放到其他路径时，才需要设置 `DECISION_AGENT_CONFIG`。

## 四、验证安装

安装 Python 3.10 或更高版本后，在项目目录中运行：

```powershell
python -m unittest discover -s tests
```

运行产生的证据和会话状态会写入项目下的 `.decision/` 目录；该目录不应随压缩包分享，也不需要手动创建。
