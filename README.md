# 🎓 Xv6 Pedagogical Agent

<div align="center">

A **teaching assistant AI** designed specifically for the xv6 operating system labs. The agent helps students learn OS concepts by providing contextual help, analyzing build errors, suggesting fixes, and guiding through implementation—all while maintaining a pedagogical approach that promotes learning over giving direct answers.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **🔍 Intelligent Help** | Context-aware assistance that understands xv6 code structure |
| **🛠️ Build Interception** | Wraps `make` to capture and analyze build errors |
| **🔧 Smart Fix Suggestions** | AI-powered error analysis with pedagogical explanations |
| **📸 Snapshot & Undo** | Session-based file tracking with rollback capability |
| **👁️ File Watching** | Real-time monitoring of code changes |
| **🎯 Task Tracking** | Progress monitoring for lab assignments |
| **🔐 Consent-Based Fixes** | Always asks before modifying your code |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **xv6 source code** (typically `xv6-public`)
- **Groq API Key** (or OpenAI/Anthropic)

### Installation

1. **Clone into your xv6 directory:**
   ```bash
   cd /path/to/xv6-public
   git clone https://github.com/your-username/xv6-agent.git .xv6_agent
   ```

2. **Run the installer:**
   ```bash
   bash .xv6_agent/install_agent.sh
   ```

3. **Activate and use:**
   ```bash
   source .venv/bin/activate
   ./agent help "How do I add a new system call?"
   ```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install tree-sitter tree-sitter-c networkx chromadb \
    sentence-transformers watchdog click rich pyyaml \
    python-dotenv groq openai anthropic

# Configure API key
echo "GROQ_API_KEY=your_key_here" > .xv6_agent/.env

# Initialize
./agent init
```

---

## 📖 Commands

### Core Commands

| Command | Description |
|---------|-------------|
| `./agent init` | Initialize the agent workspace |
| `./agent help "query"` | Get pedagogical help for your question |
| `./agent make [args]` | Run make with error interception |
| `./agent status` | Show current task/FSM state |

### Fix & Recovery

| Command | Description |
|---------|-------------|
| `./agent fix` | Analyze recent errors and propose fixes |
| `./agent fix --apply` | Apply fix immediately (turbo mode) |
| `./agent undo` | Undo the last applied change |
| `./agent snapshots` | List available file snapshots |
| `./agent restore <id>` | Restore a specific snapshot |

### Development

| Command | Description |
|---------|-------------|
| `./agent watch` | Start file watcher daemon |
| `./agent debug` | Show debug information |

---

## 💡 Usage Examples

### Getting Help
```bash
# Ask about xv6 concepts
./agent help "What is the difference between fork() and exec()?"

# Understanding code
./agent help "Explain how the scheduler works"

# Implementation guidance
./agent help "How do I implement a priority scheduler?"
```

### Fixing Build Errors
```bash
# Build with error interception
./agent make qemu

# If build fails, analyze and fix
./agent fix

# Or apply fixes automatically
./agent fix --apply
```

### Managing Changes
```bash
# View available snapshots
./agent snapshots

# Undo last change
./agent undo

# Restore specific version
./agent restore abc123
```

---

## 🏗️ Architecture

```
xv6-agent/
├── agent                    # CLI wrapper script
├── install_agent.sh         # Quick installer
├── pyproject.toml           # Python package config
└── src/xv6_agent/
    ├── cli.py               # Command-line interface
    ├── cpg_builder.py       # Code Property Graph builder
    ├── vector_store.py      # Semantic search with ChromaDB
    ├── context_assembler.py # Context preparation for LLM
    ├── llm_client.py        # LLM integration (Groq/OpenAI)
    ├── diff_engine.py       # Diff generation and application
    ├── shadow_workspace.py  # Safe testing environment
    ├── session_context.py   # Session & snapshot management
    ├── file_sentinel.py     # File change monitoring
    ├── error_recovery.py    # Error analysis & recovery
    ├── pedagogical_validator.py  # Educational response validation
    ├── task_tracker.py      # Lab progress tracking
    └── security.py          # Safety guards
```

---

## ⚙️ Configuration

### Environment Variables

Create `.xv6_agent/.env` or set in your shell:

```bash
# Required - Choose one LLM provider
GROQ_API_KEY=gsk_...
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...

# Optional
XV6_AGENT_DEBUG=false
XV6_AGENT_LOG_LEVEL=INFO
```

### Config File

Edit `.xv6_agent/config.yaml` for advanced settings:

```yaml
llm:
  provider: groq
  model: llama-3.1-70b-versatile
  
pedagogical:
  hint_level: medium      # minimal, medium, detailed
  explain_reasoning: true
  
safety:
  require_consent: true
  max_files_per_fix: 3
```

---

## 🔒 Safety Features

The agent includes multiple safety guards:

- **Consent Required**: Always asks before modifying files
- **Shadow Workspace**: Tests changes in isolation first
- **Automatic Snapshots**: Every change is reversible
- **Scope Limits**: Restricts modifications to safe boundaries
- **Pedagogical Mode**: Encourages learning over copy-paste

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- The xv6 operating system by MIT
- The pedagogical approach inspired by educational best practices
- Built with ❤️ for OS students everywhere

---

<div align="center">
<strong>Happy Learning! 🎓</strong>
</div>
