# Unity MCP

## What It Is

Unity MCP is a Model Context Protocol server that lets Claude Code interact directly with the Unity Editor. It connects the AI client to a live Editor session via a three-tier architecture:

```
Claude Code  ←→  Relay binary (~/.unity/relay/)  ←→  Unity Editor MCP Bridge
```

The relay binary is auto-installed by the `com.unity.ai.assistant` package. Claude can then inspect the project, read console errors, find assets, capture screenshots, profile runtime performance, edit scripts, and control the Editor — all from conversation.

---

## Prerequisites

- Unity 6 (6000.0+) — this project uses 6000.3.15f1 ✓
- `com.unity.ai.assistant` package installed in the Unity project
- **Unity Editor must be running** — the MCP bridge is not a background service

The relay binary installs automatically. On Windows it lives at:

```
%USERPROFILE%\.unity\relay\relay_win.exe
```

No manual configuration is needed in Claude Code — the MCP server appears automatically once the relay is installed.

---

## First Connection / Re-Approving Access

When Claude Code connects for the first time, or if the connection is ever revoked, Unity will block calls with:

```
"Connection revoked. Go to Unity Editor > Project Settings > AI > Unity MCP to change approval."
```

To fix: open Unity Editor → **Edit → Project Settings → AI → Unity MCP** → approve the pending connection. Gateway connections (Claude Code) are auto-remembered after the first approval.

---

## Verifying the Connection

With Unity Editor open and connection approved, run:

```
Unity_ManageEditor(Action="GetState")
```

A successful response looks like:

```json
{
  "success": true,
  "data": {
    "isPlaying": false,
    "isPaused": false,
    "isCompiling": false,
    "platform": "StandaloneWindows64"
  }
}
```

If it returns `"Unity not detected"`, the Editor is closed. If it returns `"Connection revoked"`, re-approve in Project Settings.

---

## Tool Reference

Tools are grouped by use case for this project.

### Project Inspection

| Tool | What it does |
|---|---|
| `Unity_GetUserGuidelines` | Load project conventions, naming rules, folder structure — call this first in any session |
| `Unity_GetProjectData` | Project overview: Unity version, installed packages, asset counts |
| `Unity_ManageEditor(Action="GetState")` | Check if Editor is playing/paused/compiling |
| `Unity_ManageEditor(Action="GetWindows")` | List currently open Editor windows |
| `Unity_ManageEditor(Action="GetTags")` | List all tags in the project |
| `Unity_ManageEditor(Action="GetLayers")` | List all layers |

### Console and Debugging

| Tool | What it does |
|---|---|
| `Unity_ReadConsole(Action="Get")` | Read console messages — supports filtering by type (Error/Warning/Log), text, timestamp |
| `Unity_ReadConsole(Action="Clear")` | Clear the console |
| `Unity_GetConsoleLogs` | Alternative console reader; simpler API |

Example — get only errors:
```
Unity_ReadConsole(Action="Get", Types=["Error"], Count=20, Format="Detailed", IncludeStacktrace=true)
```

### Asset and Script Lookup

| Tool | What it does |
|---|---|
| `Unity_FindProjectAssets(query="...")` | Semantic + name search across all project assets |
| `Unity_ListResources(Pattern="*.cs", Under="Assets/MarathonEnvs")` | List files by glob pattern under a folder |
| `Unity_FindInFile(Uri="...", Pattern="...")` | Regex search within a single file — returns line numbers |
| `Unity_Grep(args="...", path="...")` | ripgrep across project assets (`.cs` files by default) |
| `Unity_GetSha(Uri="...")` | Get SHA256 + metadata for a script without reading its contents |

Example — find all scripts that reference `BodyManager002`:
```
Unity_Grep(args="-l BodyManager002", path="Assets/MarathonEnvs")
```

### Scene and GameObject Inspection

| Tool | What it does |
|---|---|
| `Unity_ManageScene(Action="GetActive")` | Get the active scene name and path |
| `Unity_ManageScene(Action="GetHierarchy")` | Full GameObject hierarchy of the active scene |
| `Unity_ManageScene(Action="GetBuildSettings")` | List scenes in the build |
| `Unity_ManageGameObject(action="find", search_term="...")` | Find GameObjects by name |
| `Unity_ManageGameObject(action="get_components", target="...")` | List all components on a GameObject |
| `Unity_ManageGameObject(action="get_component", target="...", component_name="...")` | Read a component's properties |

Example — inspect the Walker2d agent's MarathonAgent component:
```
Unity_ManageGameObject(
  action="get_component",
  target="Walker2dAgent",
  component_name="DeepMindWalkerAgent"
)
```

### Scene Screenshots

| Tool | What it does |
|---|---|
| `Unity_SceneView_CaptureMultiAngleSceneView` | 2×2 grid: Isometric, Front, Top, Right — use to validate scene layout |
| `Unity_Camera_Capture(cameraInstanceID=...)` | Render from a specific camera by instance ID |

Note: both are computationally expensive — use sparingly.

### Script Editing

| Tool | What it does |
|---|---|
| `Unity_CreateScript(...)` | Generate a new C# script with validation |
| `Unity_ManageScript(...)` | Edit existing scripts |
| `Unity_ValidateScript(...)` | Roslyn-based syntax + dependency check (strict mode available) |
| `Unity_ApplyTextEdits(...)` | Apply text modifications to a script |
| `Unity_ScriptApplyEdits(...)` | Batch apply edits to scripts |
| `Unity_DeleteScript(...)` | Remove a script file |

### Profiling

Useful for diagnosing performance issues during ML-Agents training runs (e.g., FixedUpdate bottlenecks in `BodyManager002`).

| Tool | What it does |
|---|---|
| `Unity_Profiler_GetFrameTopTimeSam_ccc85b2d` | Top time-consuming samples for a single frame |
| `Unity_Profiler_GetFrameRangeTopTimeSummary` | Aggregated top samples across a frame range |
| `Unity_Profiler_GetFrameSelfTimeSa_e44ee448` | Self-time breakdown per sample |
| `Unity_Profiler_GetOverallGcAlloca_ac50c101` | GC allocation summary across recorded frames |
| `Unity_Profiler_GetFrameGcAllocati_a7eb5b61` | GC allocations for a specific frame |
| `Unity_Profiler_GetBottomUpSampleT_55cc1e4e` | Bottom-up call tree for a frame |

The Profiler must be recording in Unity for these to return data: **Window → Analysis → Profiler**, then press Play.

### Package Management

| Tool | What it does |
|---|---|
| `Unity_PackageManager_GetData` | List installed packages and their versions |
| `Unity_PackageManager_ExecuteAction` | Install, remove, or update packages |

### Editor Control

| Tool | What it does |
|---|---|
| `Unity_ManageEditor(Action="Play")` | Enter Play mode |
| `Unity_ManageEditor(Action="Pause")` | Pause Play mode |
| `Unity_ManageEditor(Action="Stop")` | Exit Play mode |
| `Unity_ManageEditor(Action="AddTag", TagName="...")` | Add a tag to the project |
| `Unity_ManageEditor(Action="AddLayer", LayerName="...")` | Add a layer |
| `Unity_ManageMenuItem(Action="Execute", MenuPath="...")` | Trigger any Editor menu item programmatically |
| `Unity_ManageMenuItem(Action="List", Search="...")` | Find available menu items by keyword |

Example — save the project programmatically:
```
Unity_ManageMenuItem(Action="Execute", MenuPath="File/Save Project", Refresh=false)
```

---

## Practical Workflows for This Project

### Check for ML-Agents compilation errors

```
Unity_ReadConsole(Action="Get", Types=["Error"], FilterText="MLAgents", Count=20)
```

### Find all MarathonAgent subclasses

```
Unity_Grep(args="-l MarathonAgent", path="Assets/MarathonEnvs/Agents/Scripts")
```

### Inspect the Walker2d scene hierarchy

```
Unity_ManageScene(Action="GetActive")
Unity_ManageScene(Action="GetHierarchy", Depth=2)
```

### Find Walker2d-related assets

```
Unity_FindProjectAssets(query="Walker2d")
```

### Read the reward function of DeepMindWalkerAgent

```
Unity_ListResources(Pattern="DeepMindWalkerAgent.cs", Under="Assets/MarathonEnvs")
# Then use Unity_FindInFile to search for reward logic:
Unity_FindInFile(Uri="Assets/MarathonEnvs/Agents/Scripts/DeepMindWalkerAgent.cs", Pattern="reward|Reward")
```

### Capture a screenshot of the training scene

```
# Get scene hierarchy first to find camera instance IDs:
Unity_ManageScene(Action="GetHierarchy", Depth=1)
# Then capture:
Unity_Camera_Capture(cameraInstanceID=<id>)
# Or capture multi-angle:
Unity_SceneView_CaptureMultiAngleSceneView()
```

### Profile FixedUpdate during a training session

1. Open **Window → Analysis → Profiler** in Unity Editor
2. Press Play to start a training session
3. Let it run for a few seconds, then pause
4. Call:
```
Unity_Profiler_GetFrameRangeTopTimeSummary(...)
```

---

## Limitations

- **Editor must be running** — all tools fail with `"Unity not detected"` when the Editor is closed
- **Connection must be approved** — if revoked, tools return `"Connection revoked"`; re-approve in Project Settings → AI → Unity MCP
- **Write operations are real** — `Unity_CreateScript`, `Unity_ManageScene(Action="Create")`, `Unity_ManageGameObject(action="delete")` modify the actual project; they are not dry-run
- **Screenshots are expensive** — `Unity_Camera_Capture` and `Unity_SceneView_CaptureMultiAngleSceneView` are computationally heavy; use sparingly
- **Profiler requires active recording** — profiling tools return no data unless the Unity Profiler window is open and actively recording
