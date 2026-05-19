using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Builds one executable per environment, each with envIdDefault pre-set so
/// training scripts can launch without passing --spawn-env.
///
/// Menu: Marathon Envs → Build All Environments (Windows)
///       Marathon Envs → Build Single Environment (Windows)
/// Output: <repo-root>/builds/<envId>/Marathon Environments.exe
/// </summary>
public static class BatchBuild
{
    const string ScenePath = "Assets/MarathonEnvs/Scenes/MarathonEnvs.unity";
    const string ExeName   = "Marathon Environments.exe";

    public static readonly string[] EnvIds =
    {
        "Hopper-v0",
        "Walker2d-v0",
        "Ant-v0",
        "MarathonMan-v0",
        "MarathonManSparse-v0",
        "TerrainHopper-v0",
        "TerrainWalker2d-v0",
        "TerrainAnt-v0",
        "TerrainMarathonMan-v0",
        "MarathonManWalking-v0",
        "MarathonManRunning-v0",
        "MarathonManJazzDancing-v0",
        "MarathonManMMAKick-v0",
        "MarathonManPunchingBag-v0",
        "MarathonManBackflip-v0",
        "ControllerMarathonMan-v0",
    };

    // Application.dataPath = .../UnitySDK/Assets → two levels up → repo root → builds/
    static string BuildsRoot => Path.Combine(
        Path.GetDirectoryName(Path.GetDirectoryName(Application.dataPath)),
        "builds"
    );

    // -------------------------------------------------------------------------
    // Menu items
    // -------------------------------------------------------------------------

    [MenuItem("Marathon Envs/Build All Environments (Windows)")]
    public static void BuildAll()
    {
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
            return;

        if (!EditorUtility.DisplayDialog(
            "Build All Environments",
            $"This will create {EnvIds.Length} Windows builds in:\n{BuildsRoot}\n\nThis may take a long time. Continue?",
            "Build All", "Cancel"))
            return;

        string originalEnvId = ReadEnvIdDefault();
        int succeeded = 0;

        try
        {
            for (int i = 0; i < EnvIds.Length; i++)
            {
                string envId = EnvIds[i];
                EditorUtility.DisplayProgressBar(
                    "Building Environments",
                    $"{envId}  ({i + 1} / {EnvIds.Length})",
                    (float)i / EnvIds.Length
                );

                if (BuildEnvironment(envId))
                    succeeded++;
            }
        }
        finally
        {
            EditorUtility.ClearProgressBar();
            WriteEnvIdDefault(originalEnvId);
        }

        int failed = EnvIds.Length - succeeded;
        string msg = failed == 0
            ? $"All {EnvIds.Length} environments built successfully.\n\nOutput: {BuildsRoot}"
            : $"{succeeded}/{EnvIds.Length} succeeded. {failed} failed — see Console for details.";

        EditorUtility.DisplayDialog("Build Complete", msg, "OK");
    }

    [MenuItem("Marathon Envs/Build Single Environment (Windows)")]
    static void OpenBuildSingleWindow()
    {
        BuildSingleWindow.Open();
    }

    // -------------------------------------------------------------------------
    // Core build method — public so BuildSingleWindow can call it
    // -------------------------------------------------------------------------

    public static bool BuildEnvironment(string envId)
    {
        string outputPath = Path.Combine(BuildsRoot, envId, ExeName);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

        WriteEnvIdDefault(envId);

        BuildReport report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes          = new[] { ScenePath },
            locationPathName = outputPath,
            target          = BuildTarget.StandaloneWindows64,
            options         = BuildOptions.None,
        });

        bool ok = report.summary.result == BuildResult.Succeeded;
        if (ok)
            Debug.Log($"[BatchBuild] ✓  {envId}  →  {outputPath}");
        else
            Debug.LogError($"[BatchBuild] ✗  {envId} failed  ({report.summary.totalErrors} errors)");

        return ok;
    }

    // -------------------------------------------------------------------------
    // Scene helpers — read / write envIdDefault
    // -------------------------------------------------------------------------

    public static string ReadEnvIdDefault()
    {
        var scene    = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        var selector = Object.FindFirstObjectByType<SelectEnvToSpawn>();
        return selector != null ? selector.agentSpawner.envIdDefault : "Walker2d-v0";
    }

    public static void WriteEnvIdDefault(string envId)
    {
        var scene    = EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);
        var selector = Object.FindFirstObjectByType<SelectEnvToSpawn>();
        if (selector == null)
        {
            Debug.LogError("[BatchBuild] SelectEnvToSpawn not found in scene.");
            return;
        }
        selector.agentSpawner.envIdDefault = envId;
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
    }
}

// -------------------------------------------------------------------------
// Small EditorWindow for picking a single environment to build
// -------------------------------------------------------------------------

class BuildSingleWindow : EditorWindow
{
    int _selectedIndex;

    public static void Open()
    {
        var w = GetWindow<BuildSingleWindow>("Build Single Environment");
        w.minSize = new Vector2(320, 90);
        w.ShowUtility();
    }

    void OnGUI()
    {
        EditorGUILayout.Space(8);
        _selectedIndex = EditorGUILayout.Popup("Environment", _selectedIndex, BatchBuild.EnvIds);
        EditorGUILayout.Space(8);

        if (GUILayout.Button("Build"))
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())
                return;

            string envId        = BatchBuild.EnvIds[_selectedIndex];
            string originalEnvId = BatchBuild.ReadEnvIdDefault();
            Close();

            try
            {
                BatchBuild.BuildEnvironment(envId);
            }
            finally
            {
                BatchBuild.WriteEnvIdDefault(originalEnvId);
            }
        }
    }
}
