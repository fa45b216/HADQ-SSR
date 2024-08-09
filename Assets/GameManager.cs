using UnityEngine;

public class GameManager : MonoBehaviour
{
    void Awake()
    {
        // 确保有一个 MainThreadDispatcher 实例在场景中
        if (FindObjectOfType<MainThreadDispatcher>() == null)
        {
            GameObject obj = new GameObject("MainThreadDispatcher");
            obj.AddComponent<MainThreadDispatcher>();
            DontDestroyOnLoad(obj); // 确保该对象在场景加载时不会被销毁
        }
    }
}
