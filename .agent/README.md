# Agentic Workflow Documentation

This directory contains the configuration for the specialized AI agents capable of working on the **QtOpenCVCamera** project.

## Available Agents

| Agent Name | Role | Trigger Pattern |
| :--- | :--- | :--- |
| **@[System Architect](.agent/skills/system-architect)** | High-level C++/Qt architecture, threading models, and hardware integration. | `**/*.md`, `CMakeLists.txt`, `.agent/**` |
| **@[Qt Developer](.agent/skills/qt-developer)** | QML UI implementation and C++ backend logic (signals/slots). | `content/*.qml`, `src/*.cpp`, `src/*.h` |
| **@[CV Engineer](.agent/skills/cv-engineer)** | OpenCV image processing and YOLOv8 inference optimization. | `src/inference.*`, `src/VideoController.*`, `inference/**` |
| **@[DevOps Engineer](.agent/skills/devops-engineer)** | CMake build system, dependency management, and Windows deployment. | `CMakeLists.txt`, `package.json`, `.github/**` |
| **@[QA Specialist](.agent/skills/qa-specialist)** | Functional testing, performance monitoring (FPS/RAM), and release validation. | `tests/**`, `dod.md` |
| **@[Agent Architect](.agent/skills/agent-architect)** | Manages the agent definitions and this configuration. | `.agent/**` |

## How to Use

You can direct tasks to specific agents by mentioning them in your prompt or by assigning them to specific files.

### Examples

- **UI Work**: "Ask @qt-developer to add a settings button to the main screen."
- **Computer Vision**: "Ask @cv-engineer to improve the detection confidence threshold."
- **Build Issues**: "Ask @devops-engineer to fix the linker error for ONNX Runtime."
- **Testing**: "Ask @qa-specialist to verify memory usage during long runs."
- **Architecture**: "Ask @system-architect to review the threading model for video capture."

### Agent Management
To modify the agent system itself, use the **@[Agent Architect](.agent/skills/agent-architect)**:
- **Create New Agents**: "Ask @agent-architect to create a new agent for database management."
- **Update Skills**: "Ask @agent-architect to update the @cv-engineer skill to support YOLOv9."
- **Re-evaluate Project**: "Ask @agent-architect to re-scan the codebase and update tech-stack.md."

## Configuration

- **Manifest**: `manifest.json` maps file patterns to agents.
- **Skills**: Individual agent definitions are in `skills/<agent-name>/SKILL.md`.
