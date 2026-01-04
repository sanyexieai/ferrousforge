# CI/CD 配置说明

本项目配置了完整的 CI/CD 流程，支持自动化测试、构建和发布。

## GitHub Actions

### 主要工作流

#### 1. CI (`ci.yml`)
每次推送到 `main` 或 `develop` 分支，或创建 Pull Request 时自动运行：

- **Check**: 代码格式检查和编译检查
  - `cargo fmt --all -- --check` - 检查代码格式
  - `cargo check --all-features` - 检查代码编译

- **Clippy**: 代码质量检查
  - `cargo clippy --all-features -- -D warnings` - 运行 Clippy lint

- **Test**: 运行测试套件
  - `cargo test --all-features --verbose` - 运行所有测试

- **Build**: 构建 Release 版本
  - `cargo build --release --verbose` - 构建优化版本

- **Build Matrix**: 多平台构建
  - Linux (ubuntu-latest)
  - Windows (windows-latest)
  - macOS (macos-latest)

#### 2. Release (`release.yml`)
当创建版本标签（格式：`v*.*.*`）时自动运行：

- 创建 GitHub Release
- 为多个平台构建二进制文件
- 上传构建产物到 Release

支持的平台：
- Linux x86_64
- Windows x86_64
- macOS x86_64

#### 3. CodeQL (`codeql.yml`)
代码安全扫描：

- 每周自动运行
- 在 Push 和 PR 时运行
- 使用 GitHub 的 CodeQL 进行安全分析

#### 4. Benchmark (`benchmark.yml`)
性能基准测试：

- 每周自动运行
- 在 Push 到 main 和 PR 时运行
- 目前为占位符，待实现基准测试

### Dependabot

自动依赖更新配置（`.github/dependabot.yml`）：

- 每周一自动检查依赖更新
- 自动创建 Pull Request
- 限制同时打开的 PR 数量为 10

## GitLab CI

如果使用 GitLab，项目包含 `.gitlab-ci.yml` 配置：

### 阶段

1. **Check**: 代码检查和格式化
2. **Test**: 运行测试
3. **Build**: 构建 Release 版本
4. **Release**: 发布（手动触发）

### 多平台构建

- Linux (默认)
- Windows (x86_64-pc-windows-msvc)
- macOS (x86_64-apple-darwin)

## 本地运行 CI 检查

在提交代码前，可以在本地运行 CI 检查：

```bash
# 格式化代码
cargo fmt --all

# 检查代码格式
cargo fmt --all -- --check

# 运行 Clippy
cargo clippy --all-features -- -D warnings

# 运行测试
cargo test --all-features

# 构建 Release
cargo build --release
```

## 缓存策略

CI/CD 配置了智能缓存：

- Cargo 注册表缓存
- 依赖缓存
- 构建产物缓存

缓存键基于 `Cargo.lock` 的哈希值，确保依赖更新时自动失效。

## 工作流触发条件

### GitHub Actions

- **CI**: 
  - Push 到 `main` 或 `develop`
  - 创建 Pull Request 到 `main` 或 `develop`
- **Release**: 
  - 推送版本标签（`v*.*.*`）
- **CodeQL**: 
  - Push 到 `main` 或 `develop`
  - Pull Request
  - 每周日自动运行
- **Benchmark**: 
  - Push 到 `main`
  - Pull Request 到 `main`
  - 每周日自动运行

### GitLab CI

- **Check/Test**: 
  - Merge Requests
  - Push 到 `main` 或 `develop`
- **Build**: 
  - Push 到 `main` 或 `develop`
  - 创建标签
- **Release**: 
  - 创建标签（手动触发）

## 发布流程

### 创建新版本

1. 更新 `Cargo.toml` 中的版本号
2. 提交更改
3. 创建并推送标签：
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. GitHub Actions 会自动：
   - 创建 Release
   - 构建多平台二进制文件
   - 上传到 Release

## 贡献指南

在提交代码前，请确保：

1. ✅ 代码通过 `cargo fmt` 格式化
2. ✅ 代码通过 `cargo clippy` 检查
3. ✅ 所有测试通过
4. ✅ 代码能够成功编译

## 故障排除

### CI 失败

如果 CI 失败，请检查：

1. 代码格式是否正确
2. Clippy 是否有警告
3. 测试是否全部通过
4. 构建是否成功

### 本地与 CI 不一致

如果本地通过但 CI 失败：

1. 确保使用相同的 Rust 版本
2. 清理并重新构建：`cargo clean && cargo build`
3. 检查是否有平台特定的代码

## 自定义配置

### 修改触发条件

编辑 `.github/workflows/*.yml` 中的 `on:` 部分。

### 添加新的检查

在 `ci.yml` 中添加新的 job：

```yaml
new-check:
  name: New Check
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run check
      run: cargo your-command
```

### 添加新平台

在 `build-matrix` job 中添加新的平台：

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest, new-platform]
```

## 相关文档

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [GitLab CI 文档](https://docs.gitlab.com/ee/ci/)
- [Rust 工具链文档](https://rust-lang.github.io/rustup/)

