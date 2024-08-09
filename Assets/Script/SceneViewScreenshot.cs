using UnityEditor;
using UnityEngine;

public class SceneViewScreenshot : EditorWindow
{
    private int captureWidth = 3840; // 截图宽度
    private int captureHeight = 2160; // 截图高度
    private string savePath = "Screenshots/"; // 保存路径

    [MenuItem("Tools/Scene View Screenshot")]
    public static void ShowWindow()
    {
        GetWindow<SceneViewScreenshot>("Scene View Screenshot");
    }

    private void OnGUI()
    {
        GUILayout.Label("Screenshot Settings", EditorStyles.boldLabel);
        captureWidth = EditorGUILayout.IntField("Width", captureWidth);
        captureHeight = EditorGUILayout.IntField("Height", captureHeight);
        savePath = EditorGUILayout.TextField("Save Path", savePath);

        if (GUILayout.Button("Capture Screenshot"))
        {
            CaptureScreenshot();
        }
    }

    private void CaptureScreenshot()
    {
        if (!System.IO.Directory.Exists(savePath))
        {
            System.IO.Directory.CreateDirectory(savePath);
        }

        string filePath = savePath + "Screenshot_" + System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss") + ".png";

        SceneView sceneView = SceneView.lastActiveSceneView;
        if (sceneView != null)
        {
            Camera sceneCamera = sceneView.camera;

            RenderTexture rt = new RenderTexture(captureWidth, captureHeight, 24);
            sceneCamera.targetTexture = rt;
            sceneCamera.Render();

            RenderTexture.active = rt;
            Texture2D screenshot = new Texture2D(captureWidth, captureHeight, TextureFormat.RGB24, false);
            screenshot.ReadPixels(new Rect(0, 0, captureWidth, captureHeight), 0, 0);
            screenshot.Apply();

            sceneCamera.targetTexture = null;
            RenderTexture.active = null;
            DestroyImmediate(rt);

            byte[] bytes = screenshot.EncodeToPNG();
            System.IO.File.WriteAllBytes(filePath, bytes);

            Debug.Log("Screenshot saved to: " + filePath);

            // Refresh the asset database to make the screenshot visible in the project view
            AssetDatabase.Refresh();
        }
        else
        {
            Debug.LogError("No active SceneView found!");
        }
    }
}
