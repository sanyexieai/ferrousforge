# 贡献指南

感谢您对 FerrousForge 项目的关注！我们欢迎所有形式的贡献。

## 开发环境设置

1. 克隆仓库
```bash
git clone https://github.com/your-org/ferrousforge.git
cd ferrousforge
```

2. 安装 Rust（如果还没有）
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

3. 安装依赖和工具
```bash
rustup component add rustfmt clippy
cargo build
```

## 代码风格

- 使用 `cargo fmt` 格式化代码
- 使用 `cargo clippy` 检查代码质量
- 遵循 Rust 官方编码规范

## 提交规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式（不影响代码运行）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具链相关

示例：
```
feat: add support for image models
fix: resolve memory leak in model loading
docs: update API documentation
```

## 提交流程

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## Pull Request 指南

- 确保所有测试通过
- 确保代码通过 `cargo fmt` 和 `cargo clippy` 检查
- 更新相关文档
- 添加适当的测试
- 遵循 PR 模板

## 测试

运行测试：
```bash
cargo test
```

运行特定测试：
```bash
cargo test test_name
```

## 问题报告

在创建 issue 之前，请：
- 检查是否已有类似的问题
- 使用适当的 issue 模板
- 提供足够的信息以便复现问题

## 行为准则

请遵循 [Rust 行为准则](https://www.rust-lang.org/policies/code-of-conduct)。

## 联系方式

如有问题，请通过以下方式联系：
- 创建 issue
- 发送邮件到 [2108519604@qq.com]

感谢您的贡献！

