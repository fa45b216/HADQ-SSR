using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using Unity.VisualScripting;

public class ExportPNG : MonoBehaviour
{
    // Start is called before the first frame update
    [SerializeField]
    public GameObject[] prefabs;
    void Start()
    {
        for (int i = 0; i < prefabs.Length; i++)
        {
            Debug.Log(prefabs[i].name);
            EditorUtility.SetDirty(prefabs[i]);
            Texture2D image = AssetPreview.GetAssetPreview(prefabs[i]);

            System.IO.File.WriteAllBytes("C:\\Users\\ADMIN\\Desktop\\Reinforcement_Learning\\Unity\\Project\\resource\\IMG\\" + prefabs[i].name + ".png", image.EncodeToPNG());
        }
        //for (int i = 0; i < prefabs.Length; i++)
        //{
        //    Debug.Log(prefabs[i].name);
        //    EditorUtility.SetDirty(prefabs[i]);
        //    Texture2D image = AssetPreview.GetAssetPreview(prefabs[i]);
        //    image = ResizeTexture(image, 512, 512);
        //    System.IO.File.WriteAllBytes(Application.dataPath + "/Resources/Images/" + prefabs[i].name + ".png", image.EncodeToPNG());
        //}
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
