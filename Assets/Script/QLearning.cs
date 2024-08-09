/*using UnityEngine;
using System;
using System.IO;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using Unity.VisualScripting;
using System.Reflection;
using System.Threading;
using UnityEngine.UIElements;


public class QLearning : MonoBehaviour
{
    // 三维数组表示q表，Q[x, z, action]
    private double[,,] Q;
    private Dictionary<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>> model_Q = new Dictionary<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>>();
    public bool[,] st_Q;


    //优化状态后的强化学习参数
    private Dictionary<string, Dictionary<int, double>> optimizeQ;
    private string[] optimize_previousState;
    private string[] optimize_currentState;
    private Dictionary<string, Dictionary<int, KeyValuePair<string, float>>> optimizeModelQ = new Dictionary<string, Dictionary<int, KeyValuePair<string, float>>>();

    //强化学习参数
    public float learningRate = 0.1f; // 学习率
    public float discountFactor = 0.9f; // 折扣因子
    public float explorationRate = 0.1f; // 贪婪因子
    public int Episodes = 2000;
    public int now_episode = 0;
    private float e = 2.7182818284f;
    private object qTableLock = new object(); // Q表锁
    private object modelLock = new object(); // 模型锁
    private object dataLock = new object(); // 训练数据锁


    //环境参数
    public Vector3 goalState;
    public int numActions = 8; // 动作数
    private int[] dx = { 0, -1, 0, 1, 0, 1, 1, -1, -1 };
    private int[] dz = { 1, 0, -1, 0, 0, 1, -1, 1, -1 };
    private System.Random[] random;

    //智能体参数
    public static GameObject Agent;
    private GameObject[] Agents;
    private Vector3 currentState; // 当前状态,仅表示坐标
    private Vector3 previousState; // 上一次的状态，仅表示坐标
    private float[] value1, value2; // 智能体规划的路径长度(2D, 3D)
    private int[] crash; // 发生碰撞次数
    private Vector3[] Agents_currentState;
    private Vector3[] Agents_previousState;

    private Thread[] agentThreads; // 多智能体线程
    private object agentLock = new object(); // 智能体锁
    private object taskSTLock = new object(); // 状态锁

    private bool[] taskST;

    public int Agent_num = 4;
    private float[] train_data; // 训练数据，完成训练后需要输出打印到csv文件
    private int[] train_crash;

    //训练设置参数
    [SerializeField]
    string Algorithm; // 使用的算法名称，用来命名最后输出的文件
    [SerializeField]
    bool path = false; // 是否根据训练数据画出路径曲线
    [SerializeField]
    bool dynamic = false; // 环境是否含有动态障碍物
    [SerializeField]
    QTable qTable; // 已经训练好的Q表
    [SerializeField]
    bool use_module = false; // 是否使用模型加速训练，即Dyna架构
    [SerializeField]
    private MinecraftMap map;


    // Start is called before the first frame update
    private void Start()
    {
        //map.GenerateMapRandomly(); // 随机生成三维网格地图
        //map.write(map.mapType); // 将生成的三维网格地图数据写入文件保存
        map.GenerateMapByMapType(); // 创建地图
        if (dynamic) map.GenerateObstacle();


        //优化Q表后的一些初始化工作
        optimizeQ = new Dictionary<string, Dictionary<int, double>>(); // Q表初始化
        optimize_previousState = new string[Agent_num];
        optimize_currentState = new string[Agent_num];

        //GameObject[] prefabs = map.blockPrefabs;
        //ExportPNG(prefabs);

        // 多智能体创建
        Agents = new GameObject[Agent_num];
        Agents_currentState = new Vector3[Agent_num];
        Agents_previousState = new Vector3[Agent_num];

        value1 = new float[Agent_num];
        value2 = new float[Agent_num];
        crash = new int[Agent_num];

        st_Q = new bool[map.width, map.height]; // 初始化状态表
        Q = new double[map.width, map.height, numActions]; // 初始化Q表
        for (int i = 0; i < map.width; i++)
        {
            for (int j = 0; j < map.height; j++)
            {
                for (int k = 0; k < numActions; k++)
                    Q[i, j, k] = -10000f;
            }
        }

        agentThreads = new Thread[Agent_num];
        random = new System.Random[Agent_num];
        taskST = new bool[Agent_num];
        // 初始化一些数据，具体见函数内部实现
        Init();

        train_data = new float[Episodes];
        train_crash = new int[Episodes];

        for (int i = 0; i < Episodes; i++)
        {
            train_data[i] = -1f;
        }

        for (int i = 0; i < Agent_num; i++)
        {
            int agentIndex = i;

            random[agentIndex] = new System.Random(UnityEngine.Random.Range(0, int.MaxValue));
            if (path) agentThreads[agentIndex] = new Thread(() => draw_path(agentIndex));
            else agentThreads[agentIndex] = new Thread(() => RunAgent(agentIndex));
            agentThreads[agentIndex].Start();
        }
    }

    bool checkTask()
    {
        lock (taskST)
        {
            for (int i = 0; i < Agent_num; ++i)
            {
                if (taskST[i] == false) return false;
            }
            Debug.Log("success!");
            return true;
        }
    }

   *//* private void Update()
    {

        if (path)
        {
            draw_path();
        }
        else
        {
            train();
        }
    }*//*

    // 启动agentIndex线程训练智能体
    void RunAgent(int agentIndex)
    {
        // 循环Episodes轮
        for (int episode = 0; episode < Episodes; episode++)
        {
            int ep = episode;
            //Debug.Log("agentIndex: " + agentIndex + ", episode: " + episode);
            while (!IsGoalState(Agents_currentState[agentIndex])) // 没有到达终点
            {
                Thread.Sleep(10); // 10ms,防止运行过快导致界面卡死
                int action_index;
                float Reward;
                action_index = AsyncEpsilonGreedy(ep, agentIndex, Agents_currentState[agentIndex]);
                Reward = AsyncTakeAction(agentIndex, Agents_currentState[agentIndex], action_index);
                AsyncUpdateQValue(Agents_previousState[agentIndex], action_index, Agents_currentState[agentIndex], Reward);
                //Debug.Log("agentIndex: " + agentIndex + " action: " + action_index + " reward: " + Reward);
                // 使用模型更新
                if (use_module == true)
                {
                    lock (modelLock)
                    {
                        update_model(Agents_previousState[agentIndex], action_index, Agents_currentState[agentIndex], Reward);
                        Asyncupdate_by_model(agentIndex);
                    }
                }
            }

            lock (dataLock)
            {
                if (train_data[episode] == -1f)
                {
                    train_data[episode] = value1[agentIndex];
                }
                else
                {
                    train_data[episode] = Math.Min(train_data[episode], value1[agentIndex]);// 记录本轮规划路径的代价
                }
                Debug.Log("agent: " + agentIndex + ", episode: " + episode + ", value1: " + value1[agentIndex] + ", value2: " + value2[agentIndex]);
            }
            lock (agentLock)
            {
                InitByAgentIndex(agentIndex);
            }
        }

        lock (taskSTLock)
        {
            taskST[agentIndex] = true;
        }

        if (checkTask())
        {
            WriteQtableToTxT(Q);
            WriteToCSV();
        }
    }

    void draw_path(int agentIndex)
    {
        // 循环Episodes轮
        for (int episode = 0; episode < 1; episode++)
        {
            int ep = episode;

            //Debug.Log("agentIndex: " + agentIndex + ", episode: " + episode);
            while (!IsGoalState(Agents_currentState[agentIndex])) // 没有到达终点
            {
                Thread.Sleep(500); // 10ms,防止运行过快导致界面卡死
                if (dynamic)
                {
                    MainThreadDispatcher.RunOnMainThread(() =>
                    {
                        map.dynamic_hell_move();
                    });
                }

                int action_index = AsynGetMaxQValue(Agents_currentState[agentIndex]);
                Debug.Log("action: " + action_index);
                AsyncTakeAction(agentIndex, Agents_currentState[agentIndex], action_index);
            }
        }
    }

    //void ExportPNG(GameObject[] prefabs)
    //{
    //    for (int i = 0; i < prefabs.Length; i++)
    //    {
    //        Debug.Log(prefabs[i].name);
    //        EditorUtility.SetDirty(prefabs[i]);
    //        Texture2D image = AssetPreview.GetAssetPreview(prefabs[i]);

    //        System.IO.File.WriteAllBytes("C:\\Users\\ADMIN\\Desktop\\Reinforcement_Learning\\Unity\\Project\\resource\\IMG\\" + prefabs[i].name + ".png", image.EncodeToPNG());
    //    }
    //}

    // 初始化
    void Init()
    {
        // 重新生成智能体
        for (int i = 0; i < Agent_num; i++)
        {
            value1[i] = 0;
            value2[i] = 0;
            crash[i] = 0; // 碰撞次数清空
            if (Agents[i] != null)
            {
                Destroy(Agents[i]);
            }
            Agents[i] = map.GenerateAgent();
            // 初始化每个智能体的状态
            Agents_currentState[i] = new Vector3(0f, map.mapType[0, 0] + 1.5f, 0f);
            Agents_previousState[i] = Agents_currentState[i];
        }
        if (dynamic)
        {
            map.DestroyObstacle();
            map.GenerateObstacle();
        }
        goalState = new Vector3(49f, map.mapType[map.width - 1, map.height - 1] + 1.5f, 49f); // 目标状态初始化
    }

    void InitByAgentIndex(int agentIndex)
    {
        value1[agentIndex] = 0;
        value2[agentIndex] = 0;

        MainThreadDispatcher.RunOnMainThread(() =>
        {
            if (Agents[agentIndex] != null)
            {
                Destroy(Agents[agentIndex]);
            }
            Agents[agentIndex] = map.GenerateAgent();
        });

        Agents_currentState[agentIndex] = new Vector3(0f, map.mapType[0, 0] + 1.5f, 0f);
        Agents_previousState[agentIndex] = Agents_currentState[agentIndex];

    }


    void draw_path()
    {
        currentState = Agents_currentState[0];
        previousState = Agents_previousState[0];
        if (!IsGoalState(currentState))
        {
            int action_index = choose_max(currentState);
            TakeAction(0, currentState, action_index);
        }
    }

    void train()
    {
        if (now_episode < Episodes) // 训练轮数还没结束
        {
            if (dynamic)
                map.dynamic_hell_move(); // 障碍物先移动
            for (int i = 0; i < Agent_num; i++) // 循环遍历当前智能体
            {
                currentState = Agents_currentState[i];
                previousState = Agents_previousState[i];

                if (IsGoalState(currentState)) // 到达终点
                {
                    train_data[now_episode] = value1[i]; // 记录本轮规划路径的代价
                    train_crash[now_episode] = crash[i];
                    Debug.Log("now_episode: " + now_episode + ", value1: " + value1[i] + ",value2:" + value2[i] + ",crash:" + crash[i]);
                    now_episode++;

                    Init();
                    break;
                }
                else
                {
                    int action_index = EpsilonGreedy(currentState);
                    //Debug.Log("action_index: " + action_index);
                    float Reward = TakeAction(i, currentState, action_index);
                    UpdateQValue(previousState, action_index, currentState, Reward);

                    // 使用模型更新
                    if (use_module)
                    {
                        update_model(previousState, action_index, currentState, Reward);
                        update_by_model();
                    }
                }
            }
        }
        else if (now_episode == Episodes) // 此分支只执行一次
        {
            now_episode++;
            // 训练完毕，导出需要的训练数据
            WriteToCSV();
            WriteCrashToCSV();
            WriteQtableToTxT(Q);
        }
    }

    void update_model(Vector3 state, int action, Vector3 nextState, float Reward)
    {
        if (model_Q.ContainsKey(state))
        {

            if (model_Q[state].ContainsKey(action))
            {
                model_Q[state][action] = KeyValuePair.Create(nextState, Reward);
            }
            else
            {
                model_Q[state].Add(action, KeyValuePair.Create(nextState, Reward));
            }
        }
        else
        {
            Dictionary<int, KeyValuePair<Vector3, float>> t = new Dictionary<int, KeyValuePair<Vector3, float>>
            {
                { action, KeyValuePair.Create<Vector3, float>(nextState, Reward) }
            };
            model_Q.Add(state, t);
        }

    }

    void update_by_model()
    {
        List<Vector3> mState = new List<Vector3>();
        List<KeyValuePair<Vector3, float>> mAction = new List<KeyValuePair<Vector3, float>>();
        foreach (KeyValuePair<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>> kvp in model_Q)
        {
            mState.Add(kvp.Key);
        }
        for (int _ = 0; _ < 50; _++)
        {
            int random_idx = UnityEngine.Random.Range(0, mState.Count);
            Vector3 random_state = mState[random_idx];
            random_idx = UnityEngine.Random.Range(0, model_Q[random_state].Count);
            Vector3 random_nextState = new Vector3(-1f, -1f, -1f);
            int random_action = -1;
            float reward = 0;
            foreach (KeyValuePair<int, KeyValuePair<Vector3, float>> kvp in model_Q[random_state])
            {
                if (random_idx == 0) break;
                random_action = kvp.Key;
                random_nextState = kvp.Value.Key;
                reward = kvp.Value.Value;
                random_idx--;
            }
            // void UpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)

            if (random_action != -1)
            {
                UpdateQValue(random_state, random_action, random_nextState, reward);
            }
        }
    }

    void Asyncupdate_by_model(int agentIndex)
    {
        List<Vector3> mState = new List<Vector3>();
        List<KeyValuePair<Vector3, float>> mAction = new List<KeyValuePair<Vector3, float>>();
        foreach (KeyValuePair<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>> kvp in model_Q)
        {
            mState.Add(kvp.Key);
        }
        for (int _ = 0; _ < 50; _++)
        {
            int random_idx = random[agentIndex].Next(0, mState.Count);
            Vector3 random_state = mState[random_idx];
            random_idx = random[agentIndex].Next(0, model_Q[random_state].Count);
            Vector3 random_nextState = new Vector3(-1f, -1f, -1f);
            int random_action = -1;
            float reward = 0;
            foreach (KeyValuePair<int, KeyValuePair<Vector3, float>> kvp in model_Q[random_state])
            {
                if (random_idx == 0) break;
                random_action = kvp.Key;
                random_nextState = kvp.Value.Key;
                reward = kvp.Value.Value;
                random_idx--;
            }
            // void UpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)

            if (random_action != -1)
            {
                AsyncUpdateQValue(random_state, random_action, random_nextState, reward);
            }
        }
    }

    int choose_max(Vector3 state)
    {
        return GetMaxQValue(state);
    }
    // EpsilonGreedy策略,返回选择的动作
    int EpsilonGreedy(Vector3 state)
    {
        int x = (int)state.x;
        int z = (int)state.z;

        if (st_Q[x, z] == false) // 当前状态没有探索过，所以随机选择一个动作
        {
            st_Q[x, z] = true;
            return UnityEngine.Random.Range(0, numActions);
        }

        float exp_rate = 0.0001f + (0.4f - 0.0001f) / (1 + (float)Math.Pow(e, (0.5 * (now_episode - Episodes / 2))));
        //float exp_rate = 0.1f;
        // 小概率直接随机选择动作
        if (UnityEngine.Random.Range(0f, 1f) < exp_rate)
            return UnityEngine.Random.Range(0, numActions);

        return GetMaxQValue(state);
    }

    int AsyncEpsilonGreedy(int ep, int agentIndex, Vector3 state)
    {
        lock (qTableLock)
        {
            int x = (int)state.x;
            int z = (int)state.z;

            if (st_Q[x, z] == false) // 当前状态没有探索过，所以随机选择一个动作
            {
                st_Q[x, z] = true;
                return random[agentIndex].Next(0, numActions);
            }

            float exp_rate = 0.0001f + (0.4f - 0.0001f) / (1 + (float)Math.Pow(e, (0.5 * (ep - Episodes / 2))));
            //float exp_rate = 0.1f;
            // 小概率直接随机选择动作
            if (random[agentIndex].NextDouble() < exp_rate)
                return random[agentIndex].Next(0, numActions);

            return AsynGetMaxQValue(state);
        }
    }

    void UpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)
    {
        //discountFactor = 0.9f; // 折扣因子
        //learningRate = 0.1f; // 学习率
        int nxt_x = (int)nextState.x, nxt_z = (int)nextState.z;
        //if (st_Q[nxt_x, nxt_z] == false) st_Q[nxt_x, nxt_z] = true;

        int pre_x = (int)state.x, pre_z = (int)state.z;

        int action_index = GetMaxQValue(nextState);
        double eq = Q[nxt_x, nxt_z, action_index];
        if (Q[pre_x, pre_z, action] == -1000f) Q[pre_x, pre_z, action] = 0f;
        if (eq == -1000f) eq = 0f;
        Q[pre_x, pre_z, action] += (learningRate * (Reward + discountFactor * Q[nxt_x, nxt_z, action_index] - Q[pre_x, pre_z, action]));
    }

    void AsyncUpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)
    {
        //discountFactor = 0.9f; // 折扣因子
        //learningRate = 0.1f; // 学习率
        int nxt_x = (int)nextState.x, nxt_z = (int)nextState.z;
        //if (st_Q[nxt_x, nxt_z] == false) st_Q[nxt_x, nxt_z] = true;

        int pre_x = (int)state.x, pre_z = (int)state.z;

        lock (qTableLock)
        {
            int action_index = AsynGetMaxQValue(nextState);
            Q[pre_x, pre_z, action] += (learningRate * (Reward + discountFactor * Q[nxt_x, nxt_z, action_index] - Q[pre_x, pre_z, action]));
        }

    }

    int AsynGetMaxQValue(Vector3 state)
    {
        double[,,] tmp_Q;
        if (path)
        {
            tmp_Q = qTable.Q_table;
        }
        else
        {
            tmp_Q = Q;
        }
        int x = (int)state.x, z = (int)state.z;
        int action_index = -1;
        double maxQ = 0f;
        for (int action = 0; action < numActions; action++)
        {
            if (action_index == -1)
            {
                action_index = action;
                maxQ = tmp_Q[x, z, action];
            }
            else if (maxQ < tmp_Q[x, z, action])
            {
                action_index = action;
                maxQ = tmp_Q[x, z, action];
            }
        }

        List<int> tmp = new List<int>();
        for (int i = 0; i < numActions; i++)
        {
            if (tmp_Q[x, z, i] == tmp_Q[x, z, action_index])
                tmp.Add(i);
        }

        return tmp[UnityEngine.Random.Range(0, tmp.Count)];
        return action_index;
    }

    // 返回state状态下最大Q值的动作索引
    int GetMaxQValue(Vector3 state)
    {
        double[,,] tmp_Q;
        if (path)
        {
            tmp_Q = qTable.Q_table;
        }
        else
        {
            tmp_Q = Q;
        }
        int x = (int)state.x, z = (int)state.z;
        int action_index = -1;
        double maxQ = 0f;
        for (int action = 0; action < numActions; action++)
        {
            if (action_index == -1)
            {
                action_index = action;
                maxQ = tmp_Q[x, z, action];
            }
            else if (maxQ < tmp_Q[x, z, action])
            {
                action_index = action;
                maxQ = tmp_Q[x, z, action];
            }
        }

        List<int> tmp = new List<int>();
        for (int i = 0; i < numActions; i++)
        {
            if (tmp_Q[x, z, i] == tmp_Q[x, z, action_index])
                tmp.Add(i);
        }

        return tmp[UnityEngine.Random.Range(0, tmp.Count)];
        //return action_index;
    }

    // 在执行动作内部修改状态，返回值为执行动作后获得的奖励
    // 第Agent_idx在状态state执行动作action
    // 第Agent_idx个智能体在state状态下执行动作state
    float TakeAction(int Agent_idx, Vector3 state, int action)
    {

        // 在state状态执行action动作
        int x = (int)state.x, z = (int)state.z;

        // 移动之后的x与z坐标
        x += dx[action];
        z += dz[action];

        // 移动之后未越界
        if (x >= 0 && x < map.width && z >= 0 && z < map.height && !isObstacle(x, z))
        {
            // 移动
            float y = GetCurrentStateY(new Vector3(x, 0, z));
            Vector3 next_state = new Vector3(x, y, z);
            if (Agents[Agent_idx] != null)
            {
                Destroy(Agents[Agent_idx]);
            }

            Agents[Agent_idx] = Instantiate(map.agent, next_state, Quaternion.identity);

            // 更新状态
            previousState = currentState;
            currentState = next_state;

            if (path)
            {
                //Thread.Sleep(3000);
                //Debug.DrawLine(previousState, currentState, Color.red, 100);
            }

            Agents_currentState[Agent_idx] = currentState;
            Agents_previousState[Agent_idx] = previousState;

            if (action == 4)
            {
                if (isObstacle(x + 1, z) || isObstacle(x, z + 1))
                    return 1f;
            }

            // 返回奖励
            float r = MyReward(Agent_idx);
            // float r = PeiImproveReward(Agent_idx);
            // float r = QLearningReward(Agent_idx);

            return r;
        }

        if (isObstacle(x, z)) crash[Agent_idx]++;
        // 越界返回-1的奖励
        return -1f;
    }

    bool isObstacle(int x, int z)
    {
        for (int i = 0; i < map.obstacleNum; i++)
        {
            int ox = (int)map.getObstacles()[i].transform.position.x;
            int oz = (int)map.getObstacles()[i].transform.position.z;
            if (ox == x && oz == z) return true;
        }

        return false;
    }

    float AsyncTakeAction(int Agent_idx, Vector3 state, int action)
    {
        lock (agentLock)
        {
            // 在state状态执行action动作
            int x = (int)state.x, z = (int)state.z;

            // 移动之后的x与z坐标
            x += dx[action];
            z += dz[action];

            // 移动之后未越界
            if (x >= 0 && x < map.width && z >= 0 && z < map.height)
            {
                MainThreadDispatcher.RunOnMainThread(() =>
                {
                    if (isObstacle(x, z))
                    {
                        Debug.Log("this is obstacle!!");
                    }
                });

                // 移动
                float y = GetCurrentStateY(new Vector3(x, 0, z));
                Vector3 next_state = new Vector3(x, y, z);

                MainThreadDispatcher.RunOnMainThread(() =>
                {
                    if (Agents[Agent_idx] != null)
                    {
                        Destroy(Agents[Agent_idx]);
                    }
                    Agents[Agent_idx] = Instantiate(map.agent, next_state, Quaternion.identity);
                });

                Agents_currentState[Agent_idx] = next_state;
                Agents_previousState[Agent_idx] = state;

                // 返回奖励
                float r = AsyncMyReward(Agent_idx);
                return r;
            }
            // 越界返回-1的奖励
            return -1f;
        }

    }

    float GetCurrentStateY(Vector3 state)
    {
        int x = (int)state.x, z = (int)state.z;
        int block_type = map.mapType[x, z];
        float y = block_type + 1.5f;

        return y;
    }

    bool IsGoalState(Vector3 state)
    {
        if (state == goalState)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


    float MyReward(int Agent_idx)
    {
        float w = Mathf.Abs(currentState.y - previousState.y); // 前后高度差
        value1[Agent_idx] += w + 1;
        value2[Agent_idx] += 1;
        //Debug.Log(w);
        if (IsGoalState(currentState)) return 10f;
        //return -(w + 1);

        int curr_x = (int)currentState.x, curr_z = (int)currentState.z;
        int goal_x = (int)goalState.x, goal_z = (int)goalState.z;
        float ddx = Math.Abs(curr_x - goal_x), ddz = Math.Abs(curr_z - goal_z);
        //float d = (float)Math.Sqrt(ddx * ddx + ddz * ddz);
        //float d = Math.Max(ddz, ddx); // 棋盘距离
        ddx += w; ddz += w;
        float d = (float)Math.Sqrt(ddz * ddz + ddx * ddx); // 欧氏距离
        int width = map.width + 2, height = map.height + 2;
        d = d / (float)(Math.Sqrt(width * width + height * height)); // 归一化

        float reward = 0.1f * ((float)Math.Pow(e, -2 * d));
        //Debug.Log("reward: " + reward);
        return reward;
    }

    float AsyncMyReward(int Agent_idx)
    {
        float w = Mathf.Abs(Agents_currentState[Agent_idx].y - Agents_previousState[Agent_idx].y); // 前后高度差
        value1[Agent_idx] += w + 1;
        value2[Agent_idx] += 1;
        //Debug.Log(w);
        if (IsGoalState(Agents_currentState[Agent_idx])) return 10f;
        //return -(w + 1);

        int curr_x = (int)Agents_currentState[Agent_idx].x, curr_z = (int)Agents_currentState[Agent_idx].z;
        int goal_x = (int)goalState.x, goal_z = (int)goalState.z;
        float ddx = Math.Abs(curr_x - goal_x), ddz = Math.Abs(curr_z - goal_z);
        //float d = (float)Math.Sqrt(ddx * ddx + ddz * ddz);
        //float d = Math.Max(ddz, ddx); // 棋盘距离
        ddx += w; ddz += w;
        float d = (float)Math.Sqrt(ddz * ddz + ddx * ddx); // 欧氏距离
        int width = map.width + 2, height = map.height + 2;
        d = d / (float)(Math.Sqrt(width * width + height * height)); // 归一化

        float reward = 0.1f * ((float)Math.Pow(e, -2 * d));
        //Debug.Log("reward: " + reward);
        return reward;
    }

    float PeiImproveReward(int Agent_idx)
    {
        value1[Agent_idx] += Mathf.Abs(currentState.y - previousState.y) + 1;
        value2[Agent_idx] += 1;
        if (IsGoalState(currentState)) return 5f;

        int curr_x = (int)currentState.x, curr_z = (int)currentState.z;
        int goal_x = (int)goalState.x, goal_z = (int)goalState.z;
        int ddx = Math.Abs(curr_x - goal_x), ddz = Math.Abs(curr_z - goal_z);
        float d = (float)Math.Sqrt(ddx * ddx + ddz * ddz);
        //float d = Math.Max(ddz, ddx); // 棋盘距离
        d = d / (float)(Math.Sqrt(49 * 49 * 2));
        float reward = 0.1f * ((float)Math.Pow(e, -2 * d));
        return 0.1f * ((float)Math.Pow(e, -2 * d));
    }

    float QLearningReward(int Agent_idx)
    {
        value1[Agent_idx] += Mathf.Abs(currentState.y - previousState.y) + 1;
        value2[Agent_idx] += 1;
        if (IsGoalState(currentState))
        {
            return 5f; // 到达终点
        }
        return 0f; // 没到终点
    }

    private void WriteToCSV()
    {
        string path = "C:\\Users\\ADMIN\\Desktop\\Reinforcement_Learning\\Unity\\Project\\resource\\CSV\\";//保存路径
        string fileName = path + Algorithm + "-" + DateTime.Now.ToString("yyyy-MM-dd-HH") + ".csv";//文件名
        string Datedate = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");//年月日小时分钟秒
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        if (!File.Exists(fileName))
        {
            StreamWriter sw = new StreamWriter(fileName, true, Encoding.UTF8);
            string str1 = "epsiode" + "," + "step" + "\t\n";
            sw.Write(str1);
            sw.Close();
        }

        StreamWriter swl = new StreamWriter(fileName, true, Encoding.UTF8);
        for (int episode = 0; episode < Episodes; episode++)
        {
            string str = episode + "," + train_data[episode] + "\t\n";
            swl.Write(str);
        }
        swl.Close();
    }

    private void WriteCrashToCSV()
    {
        string path = "C:\\Users\\ADMIN\\Desktop\\Reinforcement_Learning\\Unity\\Project\\resource\\CSV\\";//保存路径
        string fileName = path + Algorithm + "-Crash-" + DateTime.Now.ToString("yyyy-MM-dd-HH") + ".csv";//文件名
        string Datedate = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");//年月日小时分钟秒
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        if (!File.Exists(fileName))
        {
            StreamWriter sw = new StreamWriter(fileName, true, Encoding.UTF8);
            string str1 = "epsiode" + "," + "crash" + "\t\n";
            sw.Write(str1);
            sw.Close();
        }

        StreamWriter swl = new StreamWriter(fileName, true, Encoding.UTF8);
        for (int episode = 0; episode < Episodes; episode++)
        {
            string str = episode + "," + train_crash[episode] + "\t\n";
            swl.Write(str);
        }
        swl.Close();
    }

    private void WriteQtableToTxT(double[,,] arr)
    {
        //  Q = new float[map.width, map.height, numActions];
        string name = DateTime.Now.ToString("yyyy-MM-dd-hh-mm-s"); // 文件名字
        FileStream fs = new FileStream("D:\\" + Algorithm + "-" + name + ".txt", FileMode.Create);
        StreamWriter sw = new StreamWriter(fs);
        double[,,] s = new double[map.width, map.height, numActions];
        s = arr;

        sw.Write("{");

        for (int w = 0; w < map.width; w++)
        {
            sw.Write("{");
            for (int h = 0; h < map.height; h++)
            {
                sw.Write("{");
                for (int a = 0; a < numActions; a++)
                {
                    s[w, h, a] = arr[w, h, a];
                    sw.Write(s[w, h, a]);
                    if (a != numActions - 1)
                        sw.Write(",");
                }
                sw.Write("}");
                if (h != map.height - 1)
                    sw.Write(",");
            }
            sw.Write('}');
            if (w != map.width - 1)
                sw.Write(",");
            sw.WriteLine();
        }

        sw.Write('}');
        sw.Flush(); //清空缓冲区
        sw.Close(); //关闭流
        fs.Close();
    }

    *//*void OnApplicationQuit()
    {
        foreach (Thread thread in agentThreads)
        {
            if (thread != null && thread.IsAlive)
            {
                thread.Abort();
            }
        }
    }*//*
}


*/