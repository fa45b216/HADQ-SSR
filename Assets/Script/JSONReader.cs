using UnityEngine;
using UnityEditor;
using System.IO;

public class JSONReader : MonoBehaviour
{
    void Start()
    {
        // JSON 文件和纹理文件的路径
        string jsonDirectory = @"C:\Users\ADMIN\Desktop\Reinforcement_Learning\Unity\Project\3D_MAP\Assets\MCMaterials\Unity-1.19.4-Base-2.5.0\assets\unity\models\block\";
        string textureDirectory = @"C:\Users\ADMIN\Desktop\Reinforcement_Learning\Unity\Project\3D_MAP\Assets\MCMaterials\Unity-1.19.4-Base-2.5.0\assets\unity\textures\";

        // 获取所有 JSON 文件
        string[] jsonFiles = Directory.GetFiles(jsonDirectory, "*.json");

        foreach (string filePath in jsonFiles)
        {
            string jsonString = File.ReadAllText(filePath);
            MyJSONData jsonData = JsonUtility.FromJson<MyJSONData>(jsonString);
            CreateMaterial(jsonData, textureDirectory);
        }

        AssetDatabase.Refresh();
        Debug.Log("Materials Imported Successfully");
    }

    private void CreateMaterial(MyJSONData jsonData, string textureDirectory)
    {
        // 创建新的材质
        Material newMaterial = new Material(Shader.Find("Standard"));

        // 加载纹理
        string texturePath = Path.Combine(textureDirectory, jsonData.textures.all.Replace("unity:", "") + ".png");
        Texture2D texture = LoadTexture(texturePath);

        if (texture != null)
        {
            newMaterial.mainTexture = texture;
        }

        // 将材质保存到 Unity 项目资源中
        string savePath = "Assets/Materials/" + jsonData.textures.all.Replace("unity:", "").Replace('/', '_') + ".mat";
        AssetDatabase.CreateAsset(newMaterial, savePath);
        AssetDatabase.SaveAssets();
    }

    private Texture2D LoadTexture(string path)
    {
        if (File.Exists(path))
        {
            byte[] fileData = File.ReadAllBytes(path);
            Texture2D texture = new Texture2D(2, 2);
            texture.LoadImage(fileData); // This will auto-resize the texture dimensions.
            return texture;
        }
        Debug.LogWarning("Texture not found at path: " + path);
        return null;
    }

    [System.Serializable]
    public class MyJSONData
    {
        public string parent;
        public Textures textures;

        [System.Serializable]
        public class Textures
        {
            public string all;
        }
    }
}
