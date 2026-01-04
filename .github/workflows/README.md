# GitHub Actions 工作流说明

## 工作流文件

### 1. `ci.yml` - 持续集成
**触发条件**: Push 到 main/develop 或创建 PR

**包含任务**:
- ✅ 代码格式检查 (`cargo fmt`)
- ✅ 代码编译检查 (`cargo check`)
- ✅ 代码质量检查 (`cargo clippy`)
- ✅ 运行测试 (`cargo test`)
- ✅ 构建 Release (`cargo build --release`)
- ✅ 多平台构建（Linux/Windows/macOS）

### 2. `release.yml` - 发布流程
**触发条件**: 推送版本标签（如 `v0.1.0`）

**功能**:
- 自动创建 GitHub Release
- 构建多平台二进制文件
- 上传构建产物到 Release

### 3. `codeql.yml` - 安全扫描
**触发条件**: Push/PR 或每周日自动运行

**功能**:
- 使用 GitHub CodeQL 进行代码安全分析
- 检测潜在的安全漏洞

### 4. `benchmark.yml` - 性能基准测试
**触发条件**: Push 到 main/PR 或每周日自动运行

**功能**:
- 运行性能基准测试（待实现）

## 使用说明

### 本地运行 CI 检查

```bash
# 格式化代码
cargo fmt --all

# 检查格式
cargo fmt --all -- --check

# 运行 Clippy
cargo clippy --all-features -- -D warnings

# 运行测试
cargo test --all-features

# 构建 Release
cargo build --release
```

### 创建新版本

```bash
# 1. 更新 Cargo.toml 版本号
# 2. 提交更改
git commit -am "chore: bump version to 0.1.0"

# 3. 创建并推送标签
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions 会自动创建 Release 并上传构建产物。

## 缓存

所有工作流都配置了智能缓存：
- Cargo 注册表缓存
- 依赖缓存
- 构建产物缓存

缓存键基于 `Cargo.lock`，确保依赖更新时自动失效。

## 故障排除

如果 CI 失败：
1. 检查代码格式：`cargo fmt --all -- --check`
2. 检查 Clippy：`cargo clippy --all-features`
3. 运行测试：`cargo test --all-features`
4. 检查构建：`cargo build --release`

