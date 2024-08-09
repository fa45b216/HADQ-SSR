using UnityEngine;

public class GameManager : MonoBehaviour
{
    void Awake()
    {
        // ȷ����һ�� MainThreadDispatcher ʵ���ڳ�����
        if (FindObjectOfType<MainThreadDispatcher>() == null)
        {
            GameObject obj = new GameObject("MainThreadDispatcher");
            obj.AddComponent<MainThreadDispatcher>();
            DontDestroyOnLoad(obj); // ȷ���ö����ڳ�������ʱ���ᱻ����
        }
    }
}
