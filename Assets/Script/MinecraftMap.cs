using System;
using System.Collections.Generic;
using System.IO;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UIElements;

public class MinecraftMap : MonoBehaviour
{
    public int width = 50;
    public int height = 50;
    public int obstacleNum = 30;
    public int startIndex = 10;
    public int maxHeight = 1;
    public GameObject[] blockPrefabs; // Different types of blocks
    public GameObject[,,] blocksGrid;
    public float[] h = new float[3]{ 0.5f, 1.0f, 1.5f };
    public int[,] mapType; // 存储随机生成的地图类型
    [SerializeField]
    private MapType map; // 已经生成的地图类型
    [SerializeField]
    public GameObject agent; // 智能体
    public GameObject Preagent;
    [SerializeField]
    public GameObject PrefabObstacle;
    private GameObject[] Obstacles = new GameObject[30];
    public int[] ObstaclesSpeed;
    public int[] ObstaclesState;
    public int[] Obstaclesincrement;
    

    private void Start()
    {
        Obstacles = new GameObject[obstacleNum];
        ObstaclesSpeed = new int[obstacleNum];
        ObstaclesState = new int[obstacleNum];
        Obstaclesincrement = new int[obstacleNum];
        // 初始化障碍物移动速度
        for (int i = 0; i < obstacleNum; i ++)
        {
            ObstaclesSpeed[i] = (i % 5) + 1;
            ObstaclesState[i] = 0;
            Obstaclesincrement[i] = 1;
        }
    }
    //public Dictionary<string, int> keyValuePairs = new Dictionary<string, int>();

    // 通过已有的地图生成地图
    public void GenerateMapByMapType()
    {
        int c0 = 0, c1 = 0, c2 = 0;
        mapType = map.maptype;
        blocksGrid = new GameObject[width, maxHeight, height];
        for (int x = 0; x < width; x ++)
        {
            for (int z = 0; z < height; z ++)
            {
                int blockType = mapType[x, z];
                if (blockType == 0) c0++;
                else if (blockType == 1) c1++;
                else if (blockType == 2) c2++;
                //Vector3 spawnPosition = new Vector3(x, h[blockType], z);
                Vector3 spawnPosition = new Vector3(x, 0, z);
                GameObject block = Instantiate(blockPrefabs[blockType], spawnPosition, Quaternion.identity);
                blocksGrid[x, 0, z] = block;
            }
        }
    }

    // 创建智能体
    public GameObject GenerateAgent()
    {
        int block_type = mapType[0, 0];
        float y = block_type + 1.5f;
        Vector3 spawnAgent = new Vector3(0, y, 0);
        Preagent = Instantiate(agent, spawnAgent, Quaternion.identity);
        return Preagent;
    }

    public void GenerateObstacle()
    {
        int cnt = 0;
        for (int i = startIndex; i < startIndex + obstacleNum; i += 1)
        {
            float y = mapType[0, i] + 1.5f;
            Vector3 pos = new Vector3(0, y, i); // 坐标
            Obstacles[cnt++] = Instantiate(PrefabObstacle, pos, Quaternion.identity);
        }
    }


    public void DestroyObstacle()
    {
        for (int i = 0; i < obstacleNum; i += 1)
        {
            if (Obstacles[i] != null)
            {
                Destroy(Obstacles[i]);
            }
        }
    }

    // 动态障碍物的移动
    public void dynamic_hell_move()
    {
        for (int i = 0; i < obstacleNum; i ++)
        {
            ObstaclesState[i] += 1;
            /*if (Obstacles[i] == null)
            {
                Debug.Log(i + " " + "is null !!");
                continue;
            }*/
            if (ObstaclesState[i] % ObstaclesSpeed[i] == 0)
            {
                ObstaclesState[i] = 0;
                float x = Obstacles[i].transform.position.x;
                float z = Obstacles[i].transform.position.z;
                if (x + Obstaclesincrement[i] >= width) Obstaclesincrement[i] = -1;
                else if (x + Obstaclesincrement[i] < 0) Obstaclesincrement[i] = 1;
                x += Obstaclesincrement[i];
                Destroy(Obstacles[i]);

                Vector3 pos = new Vector3 (x, mapType[(int)x, (int)z] + 1.5f, z);
                Obstacles[i] = Instantiate(PrefabObstacle, pos, Quaternion.identity);
            }
        }
    }
    

    public GameObject[] getObstacles()
    {
        return Obstacles;
    }

    // 随机生成地图
    public void GenerateMapRandomly()
    {
        mapType = new int[width, height];
        blocksGrid = new GameObject[width, maxHeight, height];
        for (int x = 0; x < width; x++)
        {
            for (int z = 0; z < height; z++)
            {
                int blockType = UnityEngine.Random.Range(0, blockPrefabs.Length);
                mapType[x, z] = blockType;
                Vector3 spawnPosition = new Vector3(x, h[blockType], z);
                GameObject block = Instantiate(blockPrefabs[blockType], spawnPosition, Quaternion.identity);
                blocksGrid[x, 0, z] = block;
            }
        }
    }


    // 存储随机生成的地图模型
    public void write(int[,] arr)
    {
        string name = DateTime.Now.ToString("yyyy-MM-dd-hh-mm-ss");
        // C:\Users\ADMIN\Desktop\Reinforcement_Learning\Unity\Project\resource\MAP
        FileStream fs = new FileStream("C:\\Users\\ADMIN\\Desktop\\Reinforcement_Learning\\Unity\\Project\\resource\\MAP\\" + name + ".txt", FileMode.Create);
        StreamWriter sw = new StreamWriter(fs);
        int[,] s = new int[width, height];
        s = arr;

        sw.Write("{");
        for (int x = 0; x < width; x++)
        {
            sw.Write("{");
            for (int z = 0; z < height; ++z)
            {
                s[x, z] = arr[x, z];
                int output;
                output = Convert.ToInt32(s[x, z]);
                sw.Write(output);
                if (z != height - 1)
                    sw.Write(",");
            }
            sw.Write("}");
            if (x != width - 1)
                sw.Write(",");
            sw.WriteLine();
        }
        sw.Write("}");
        sw.Flush(); //清空缓冲区
        sw.Close(); //关闭流
        fs.Close();
    }
}