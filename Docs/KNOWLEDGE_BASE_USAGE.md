# Knowledge Base Selection Guide

This guide explains how to use the enhanced knowledge base selection features in Synapse.

## Overview

After running `python initialize_fast.py`, you'll have multiple specialized knowledge bases, one for each folder in your `Data/` directory. You can now select which knowledge base to use for your queries.

## Available Commands

### 1. List Available Knowledge Bases

```bash
# Command line query tool
python pipeline/query.py --list-kb

# Interactive chat
python chat.py --list-kb
```

### 2. Select Knowledge Base by Name

```bash
# Command line query tool
python pipeline/query.py --kb "Aahan_s_Notes" --question "What is RISC-V?"
python pipeline/query.py --kb "hash_Confluence_IPS" --question "IPS requirements"

# Interactive chat
python chat.py --kb "Aahan_s_Notes"
```

### 3. Interactive Knowledge Base Selection

```bash
# Command line query tool
python pipeline/query.py --select-kb --question "Your question here"

# Interactive chat
python chat.py --select-kb
```

## Interactive Chat Features

When using `python chat.py`, you have additional commands:

- `/kb` - List available knowledge bases and show which one is currently active
- `/switch-kb` - Interactively switch to a different knowledge base mid-conversation
- `/help` - Show all available commands

## GUI Interface

The web GUI also supports knowledge base selection:

```bash
python chat.py --test_gui --kb "Aahan_s_Notes"
```

The GUI includes API endpoints for:
- `/api/knowledge-bases` - List available knowledge bases
- `/api/switch-kb` - Switch knowledge bases dynamically

## Knowledge Base Naming

Knowledge bases are created based on your folder structure:

- Regular folders: `Aahan_s_Notes`, `Ascalon_Docs`
- Folders starting with `#`: `hash_Confluence_IPS`, `hash_Confluence_PSE`

Display names show friendly versions: `Aahan's Notes`, `#Confluence/IPS`

## Examples

### Example 1: Query specific documentation
```bash
# Query Ascalon documentation
python pipeline/query.py --kb "Ascalon_Docs" --question "What are the Ascalon CPU features?"

# Query Confluence IPS data
python pipeline/query.py --kb "hash_Confluence_IPS" --question "What are the IPS requirements?"
```

### Example 2: Interactive session with knowledge base switching
```bash
python chat.py --select-kb
# Select knowledge base interactively
# Ask questions
# Use /switch-kb to change knowledge bases during conversation
```

### Example 3: GUI with specific knowledge base
```bash
python chat.py --test_gui --kb "Aahan_s_Notes"
```

## Benefits

1. **Focused Results**: Query specific domains for more relevant results
2. **Reduced Noise**: Avoid irrelevant information from other folders  
3. **Better Performance**: Smaller knowledge bases = faster retrieval
4. **Domain Expertise**: Each knowledge base specializes in its content area
5. **Dynamic Switching**: Change knowledge bases without restarting

## Tips

- Use `--list-kb` first to see what knowledge bases are available
- Knowledge base names are case-sensitive
- The interactive selector shows chunk counts and file sizes
- You can switch knowledge bases mid-conversation in chat mode
- GUI interface preserves your knowledge base selection across queries
