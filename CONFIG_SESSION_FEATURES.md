# Config-to-Session Linking Features

## Overview
Successfully implemented comprehensive config-to-session linking functionality that allows users to efficiently experiment with configuration parameters while maintaining separate configs per session.

## Features Implemented

### 1. Default Configuration Management
- **File**: `default_config.json` - Stores the default configuration template
- **Functions**: `load_default_config()` and `save_default_config()` in `chat.py`
- **Purpose**: Provides a persistent default configuration that new sessions inherit

### 2. Session-Specific Configuration Storage
- **Enhanced ChatSession Class**: Now stores and manages configuration per session
- **Session Format**: Extended to include `config` field alongside `history`
- **Auto-save**: Configurations are automatically saved with session data

### 3. Flask API Endpoints
- `GET /api/config` - Get current session configuration
- `POST /api/config` - Update current session configuration  
- `GET /api/config/default` - Get default configuration
- `POST /api/config/save-as-default` - Save current config as new default
- `POST /api/config/reset-to-default` - Reset current session to default config

### 4. Enhanced UI Components
- **Save as Default Button** (⭐) - Saves current session config as the new default
- **Reset to Default Button** (🔄) - Resets current session to default values
- **Config Status Indicator** - Shows config state and session linkage
- **Session Tips** - User guidance about per-session configurations

### 5. Intelligent Session Management
- **New Sessions**: Automatically inherit current default configuration
- **Session Loading**: Preserves and loads session-specific configurations
- **Config Switching**: UI automatically updates when switching between sessions
- **Fallback Handling**: Graceful fallback to defaults for legacy sessions

## User Workflow

### Experimenting with Configurations
1. **Modify Parameters**: Adjust system prompt, top-k chunks, MMR lambda, etc.
2. **Auto-save**: Changes are automatically saved to current session
3. **Test & Iterate**: Ask questions to see how config changes affect responses
4. **Save Successful Config**: Use "Save as Default" to preserve good configurations

### Session Management
1. **Per-Session Configs**: Each session maintains its own configuration
2. **Config Inheritance**: New sessions start with current default config
3. **Session Switching**: Switching sessions automatically loads their configs
4. **Config Reset**: Reset any session back to default values anytime

### Default Configuration Management
1. **Save as Default**: Promote any session's config to be the new default
2. **Persistent Storage**: Default configs persist across application restarts
3. **New Session Template**: New sessions automatically use the saved default

## Technical Implementation

### Backend (chat.py)
- Extended `ChatSession` class with config management methods
- Added config persistence to session JSON files
- Implemented config loading/saving utilities
- Updated Flask endpoints for config management

### Frontend (main.js)
- Added config status tracking and UI updates
- Implemented save-as-default and reset-to-default functionality
- Enhanced session loading to handle config switching
- Added user feedback for config operations

### UI (index.html)
- Added Config Management section with new buttons
- Included helpful tooltips and status indicators
- Added user tips for session-config relationship

## Configuration Parameters
All standard parameters are session-linked:
- `system_prompt` - Custom AI behavior instructions
- `topk` - Number of context chunks (1-20)
- `per_doc` - Maximum chunks per document (1-12)
- `lambda_mmr` - MMR diversity parameter (0-1)
- `timeout` - LLM timeout in seconds (10-180)
- `verbose` - Detailed retrieval logging
- `max_images` - Maximum images per response (0-10)
- `images_enabled` - Enable/disable image retrieval

## Benefits for Users

### Experimentation Efficiency
- **No Loss of Work**: Experiment freely without losing good configurations
- **Quick Switching**: Switch between different config setups via sessions
- **Iterative Improvement**: Build up library of specialized configurations

### Session Context
- **Purpose-Specific Configs**: Different sessions for different use cases
- **Preserved Settings**: Session configs persist across app restarts
- **Clear Feedback**: Always know which config is active

### Team Collaboration
- **Shared Defaults**: Team members can share and use the same defaults
- **Personal Sessions**: Individual team members can have personal configs
- **Best Practice Sharing**: Save proven configurations as defaults

## Files Modified

1. **chat.py**: Enhanced ChatSession class, added config management functions, updated Flask endpoints
2. **templates/index.html**: Added Config Management UI section with new buttons and indicators
3. **static/js/main.js**: Implemented frontend config management logic and session switching
4. **default_config.json**: New file for storing default configuration template

## Testing Verification

The implementation has been tested for:
- ✅ Config persistence per session
- ✅ Default config saving and loading
- ✅ Session switching with config preservation
- ✅ UI feedback and status updates
- ✅ Graceful handling of legacy sessions
- ✅ Auto-save functionality
- ✅ Reset to defaults functionality

## Usage Examples

### Example 1: Research Session
1. Create session for research work
2. Set high top-k (15) and verbose mode for detailed analysis
3. Use specialized system prompt for technical documentation
4. Save config to session automatically

### Example 2: Quick Queries
1. Create session for quick questions
2. Set low top-k (5) and fast timeout for speed
3. Use concise system prompt
4. If this works well, save as default

### Example 3: Image Analysis
1. Create session for visual content analysis
2. Enable images, set max_images to 5
3. Adjust system prompt for image description tasks
4. Switch to other sessions without losing this setup

The system now provides a seamless experience for configuration experimentation while maintaining the context and settings that work best for different types of tasks.
