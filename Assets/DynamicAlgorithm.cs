using UnityEngine;
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
using static UnityEngine.Random;
using System.Xml;
using System.Runtime.ConstrainedExecution;
using static UnityEngine.UIElements.UxmlAttributeDescription;


[System.Serializable]
public class Q_Table
{
    public List<QState> states = new List<QState>();
}

[System.Serializable]
public class QState
{
    public string state;
    public List<QAction> actions = new List<QAction>();
}

[System.Serializable]
public class QAction
{
    public int action;
    public double value;
}

public class DynamicAlgorithm : MonoBehaviour
{
    // ��ά�����ʾq��Q[x, z, action]
    private double[,,] Q;
    private Dictionary<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>> model_Q = new Dictionary<Vector3, Dictionary<int, KeyValuePair<Vector3, float>>>();
    public bool[,] st_Q;


    /*�Ż�״̬���ǿ��ѧϰ����*/
    private Dictionary<string, Dictionary<int, double>> optimizeQ;

    private string[] optimize_previousState;
    private string[] optimize_currentState;
    private Dictionary<string, Dictionary<int, KeyValuePair<string, float>>> optimizeModelQ = new Dictionary<string, Dictionary<int, KeyValuePair<string, float>>>();

    /*ǿ��ѧϰ����*/
    public float learningRate = 0.1f; // ѧϰ��
    public float discountFactor = 0.9f; // �ۿ�����
    public float explorationRate = 0.1f; // ̰������
    public int Episodes = 2000;
    public int now_episode = 0;
    private float e = 2.7182818284f;
    private object qTableLock = new object(); // Q����
    private object modelLock = new object(); // ģ����
    private object dataLock = new object(); // ѵ��������


    /*��������*/
    public Vector3 goalState;
    public int numActions = 8; // ������
    private int[] dx = { 0, -1, 0, 1, 0, 1, 1, -1, -1 };
    private int[] dz = { 1, 0, -1, 0, 0, 1, -1, 1, -1 };

    private int[] sdx = { -1, 0, 1, 1, 1, 0, -1, -1};
    private int[] sdz = { 1, 1, 1, 0, -1, -1, -1, 0};
    private System.Random[] random;

    /*���������*/
    public static GameObject Agent;
    public bool traditionalState = false;
    private GameObject[] Agents;
    private Vector3 currentState; // ��ǰ״̬,����ʾ����
    private Vector3 previousState; // ��һ�ε�״̬������ʾ����
    private float[] value1, value2; // ������滮��·������(2D, 3D)
    private int[] crash; // ������ײ����
    private string[] Agents_currentState;
    private string[] Agents_previousState;

    private Thread[] agentThreads; // ���������߳�
    private object agentLock = new object(); // ��������
    private object taskSTLock = new object(); // ״̬��

    private bool[] taskST;

    public int Agent_num = 4;
    private float[] train_data; // ѵ�����ݣ����ѵ������Ҫ�����ӡ��csv�ļ�
    private int[] train_crash;

    /*ѵ�����ò���*/
    [SerializeField]
    string Algorithm; // ʹ�õ��㷨���ƣ������������������ļ�
    [SerializeField]
    bool path = false; // �Ƿ����ѵ�����ݻ���·������
    [SerializeField]
    bool dynamic = false; // �����Ƿ��ж�̬�ϰ���
    [SerializeField]
    QTable qTable; // �Ѿ�ѵ���õ�Q��
    [SerializeField]
    bool use_module = false; // �Ƿ�ʹ��ģ�ͼ���ѵ������Dyna�ܹ�
    [SerializeField]
    private MinecraftMap map;

    [SerializeField]
    public DrawLineBetweenPoints brush;

    List<Vector3> points = new List<Vector3>();

    // Start is called before the first frame update
    private void Start()
    {

        //map.GenerateMapRandomly(); // ���������ά�����ͼ
        //map.write(map.mapType); // �����ɵ���ά�����ͼ����д���ļ�����
        map.GenerateMapByMapType(); // ������ͼ
        if (dynamic) map.GenerateObstacle();

        if (path) Agent_num = 1;
        /*�Ż�Q����һЩ��ʼ������*/
        optimizeQ = new Dictionary<string, Dictionary<int, double>>(); // Q���ʼ��
        optimize_previousState = new string[Agent_num];
        optimize_currentState = new string[Agent_num];

        //GameObject[] prefabs = map.blockPrefabs;
        //ExportPNG(prefabs);

        // �������崴��
        Agents = new GameObject[Agent_num];
        Agents_currentState = new string[Agent_num];
        Agents_previousState = new string[Agent_num];

        value1 = new float[Agent_num];
        value2 = new float[Agent_num];
        crash = new int[Agent_num];

        st_Q = new bool[map.width, map.height]; // ��ʼ��״̬��
        Q = new double[map.width, map.height, numActions]; // ��ʼ��Q��
        /*for (int i = 0; i < map.width; i ++)
        {
            for (int j = 0; j < map.height; j ++)
            {
                for (int k = 0; k < numActions; k++)
                    Q[i, j, k] = -10000f;
            }
        }*/

        agentThreads = new Thread[Agent_num];
        random = new System.Random[Agent_num];
        taskST = new bool[Agent_num];
        // ��ʼ��һЩ���ݣ�����������ڲ�ʵ��
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
            //else agentThreads[agentIndex] = new Thread(() => RunAgent(agentIndex));
            agentThreads[agentIndex].Start();
        }

    }

    private void Update()
    {
       //train();
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

    // ��ʼ��
    void Init()
    {
        // ��������������
        for (int i = 0; i < Agent_num; i++)
        {
            value1[i] = 0;
            value2[i] = 0;
            crash[i] = 0; // ��ײ�������
            if (Agents[i] != null)
            {
                Destroy(Agents[i]);
            }
            Agents[i] = map.GenerateAgent();
            // ��ʼ��ÿ���������״̬
            string state = getState(0, 0);

            Agents_currentState[i] = state;
            Agents_previousState[i] = Agents_currentState[i];
        }

        // ������̬�ϰ���
        if (dynamic)
        {
            map.DestroyObstacle();
            map.GenerateObstacle();
        }
        goalState = new Vector3(49f, map.mapType[map.width - 1, map.height - 1] + 1.5f, 49f); // Ŀ��״̬��ʼ��
    }

    string getState(int x, int z)
    {
        string state = Convert.ToString(x) + "-" + Convert.ToString(z) + "-";

        if (!traditionalState) // ���ô�ͳ״̬��ʾ����
        {
            for (int j = 0; j < 8; j++)
            {
                int ex = x + sdx[j], ez = z + sdz[j];

                if (ex < 0 || ex >= map.width || ez < 0 || ez >= map.height) state += "2-";
                else if (isObstacle(ex, ez))
                {
                    int v = -1;
                    for (int k = 0; k < map.obstacleNum; k++)
                    {
                        if (ex == map.getObstacles()[k].transform.position.x && ez == map.getObstacles()[k].transform.position.z)
                        {
                            v = k;
                            break;
                        }
                    }
                    state += Convert.ToString(map.Obstaclesincrement[v]) + "-";
                }
                else state += "0-";
            }
        }
        return state;
    }

    /*void InitByAgentIndex(int agentIndex)
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

    }*/


    void draw_path(int agentIndex)
    {
        Thread.Sleep(5000);
        Q_Table qtable = LoadQTableFromFile("D:\\Heuristic-Asynchronous-Dyna-Q-crash2000-5-2024-06-05-07-51-0");
        if (qtable != null)
        {
            optimizeQ = ConvertFromQTable(qtable);
        }
        else
        {
            Debug.Log("qtable is null!!!");
        }
        Vector3 cur = new Vector3 (0, 0, 0);
        MainThreadDispatcher.RunOnMainThread(() =>
        {
            cur = Agents[agentIndex].transform.position;
        });
        points.Add(new Vector3(0, 4, 0));
        while (!IsGoalState(cur)) // û�е����յ�
        {
            Thread.Sleep(250);
            if (dynamic)
            {
                MainThreadDispatcher.RunOnMainThread(() =>
                {
                    map.dynamic_hell_move();
                });
            }

            int action_index = GetMaxQValue(Agents_currentState[agentIndex]);
            if (action_index == -1) break;
            Debug.Log("action: " + action_index);
            MainThreadDispatcher.RunOnMainThread(() =>
            {
                TakeAction(agentIndex, action_index);
                brush.DrawLineByPoints(points);
                cur = Agents[agentIndex].transform.position;
            });
        }
    }

    void train()
    {
        if (now_episode < Episodes) // ѵ��������û����
        {

            /*for (int i = 0; i < Agent_num; i ++)
            {
                Agents_currentState[i] = getState((int)Agents[i].transform.position.x, (int)Agents[i].transform.position.z);
            }*/

            if (dynamic)
                map.dynamic_hell_move(); // �ϰ������ƶ�

            for (int i = 0; i < Agent_num; i++) // ѭ��������ǰ������
            {
                if (IsGoalState(Agents[i].transform.position)) // �����յ�
                {
                    train_data[now_episode] = value1[i]; // ��¼���ֹ滮·���Ĵ���
                    train_crash[now_episode] = crash[i];
                    Debug.Log("now_episode: " + now_episode + ", value1: " + value1[i] + ",value2:" + value2[i] + ",crash:" + crash[i]);
                    now_episode++;

                    Init();
                    break;
                }
                else
                {
                    // ���������������ѡ����
                    int action_index = EpsilonGreedy(i);
                    //Debug.Log("action_index: " + action_index);
                    // ִ�ж���
                    previousState = Agents[i].transform.position;
                    float Reward = TakeAction(i, action_index);
                    currentState = Agents[i].transform.position;

                    // ����Qֵ
                    UpdateQValue(i, action_index, Reward);

                    // ʹ��ģ�͸���
                    if (use_module)
                    {
                        update_model(i, action_index, Reward);
                        update_by_model();
                    }
                }
            }
        }
        else if (now_episode == Episodes) // �˷�ִֻ֧��һ��
        {
            now_episode++;
            // ѵ����ϣ�������Ҫ��ѵ������
            WriteToCSV();
            WriteCrashToCSV();
            /*WriteQtableToTxT(Q);*/
            SaveQTableToFile(optimizeQ);
        }
    }

    void update_model(int agentIndex, int action, float Reward)
    {
        string state = Agents_previousState[agentIndex];
        string nextState = Agents_currentState[agentIndex];
        if (optimizeModelQ.ContainsKey(state))
        {
            if (optimizeModelQ[state].ContainsKey(action))
            {
                optimizeModelQ[state][action] = KeyValuePair.Create(nextState, Reward);
            }
            else
            {
                optimizeModelQ[state].Add(action, KeyValuePair.Create(nextState, Reward));
            }
        }
        else
        {
            Dictionary<int, KeyValuePair<string, float>> t = new Dictionary<int, KeyValuePair<string, float>>
            {
                { action, KeyValuePair.Create<string, float>(nextState, Reward) }
            };
            optimizeModelQ.Add(state, t);
        }

    }

    void update_by_model()
    {
        List<string> mState = new List<string>();
        List<KeyValuePair<string, float>> mAction = new List<KeyValuePair<string, float>>();

        // ��ȡ����״̬
        foreach (KeyValuePair<string, Dictionary<int, KeyValuePair<string, float>>> kvp in optimizeModelQ)
        {
            mState.Add(kvp.Key);
        }

        // ����50��
        for (int _ = 0; _ < 50; _++)
        {
            // ���ѡ��״̬
            int random_idx = UnityEngine.Random.Range(0, mState.Count);
            string random_state = mState[random_idx];

            random_idx = UnityEngine.Random.Range(0, optimizeModelQ[random_state].Count);
            string random_nextState = "";
            int random_action = -1;
            float reward = 0;
            foreach (KeyValuePair<int, KeyValuePair<string, float>> kvp in optimizeModelQ[random_state])
            {
                random_action = kvp.Key;
                random_nextState = kvp.Value.Key;
                reward = kvp.Value.Value;
                if (random_idx == 0) break;
                random_idx--;
            }
            // void UpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)

            if (random_action != -1)
            {
                UpdateModelQValue(random_state, random_action, reward, random_nextState);
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

    /*int choose_max(Vector3 state)
    {
        return GetMaxQValue(state);
    }*/

    // EpsilonGreedy����,����ѡ��Ķ���
    int EpsilonGreedy(int agentIndex)
    {
        // ��ǰ״̬û��̽�������������ѡ��һ������
        if (optimizeQ.ContainsKey(Agents_currentState[agentIndex]) == false)
        {
            optimizeQ.Add(Agents_currentState[agentIndex], new Dictionary<int, double>());
            //optimizeQ.Add(Agents_currentState[agentIndex], );
            return UnityEngine.Random.Range(0, numActions);
        }

        //float exp_rate = 0.0001f + (0.4f - 0.0001f) / (1 + (float)Math.Pow(e, (0.5 * (now_episode - Episodes / 2))));
        float exp_rate = 0.1f;
        // С����ֱ�����ѡ����
        if (UnityEngine.Random.Range(0f, 1f) < exp_rate)
            return UnityEngine.Random.Range(0, numActions);
        if (optimizeQ[Agents_currentState[agentIndex]].Count == 0) return UnityEngine.Random.Range(0, numActions);
        else return GetMaxQValue(Agents_currentState[agentIndex]);
    }

    int AsyncEpsilonGreedy(int ep, int agentIndex, Vector3 state)
    {
        lock (qTableLock)
        {
            int x = (int)state.x;
            int z = (int)state.z;

            if (st_Q[x, z] == false) // ��ǰ״̬û��̽�������������ѡ��һ������
            {
                st_Q[x, z] = true;
                return random[agentIndex].Next(0, numActions);
            }

            float exp_rate = 0.0001f + (0.4f - 0.0001f) / (1 + (float)Math.Pow(e, (0.5 * (ep - Episodes / 2))));
            //float exp_rate = 0.1f;
            // С����ֱ�����ѡ����
            if (random[agentIndex].NextDouble() < exp_rate)
                return random[agentIndex].Next(0, numActions);

            return AsynGetMaxQValue(state);
        }
    }

    void UpdateQValue(int agentIndex, int action, float Reward)
    {
        int action_index = GetMaxQValue(Agents_currentState[agentIndex]);
        double maxNxtQ = 0;
        if (action_index != -1)
        {
            maxNxtQ = optimizeQ[Agents_currentState[agentIndex]][action_index];
        }

        if (optimizeQ[Agents_previousState[agentIndex]].ContainsKey(action) == false)
        {
            optimizeQ[Agents_previousState[agentIndex]].Add(action, 0);
        }
        optimizeQ[Agents_previousState[agentIndex]][action] += (learningRate * (Reward + discountFactor * maxNxtQ - optimizeQ[Agents_previousState[agentIndex]][action]));
        //Q[pre_x, pre_z, action] += (learningRate * (Reward + discountFactor * Q[nxt_x, nxt_z, action_index] - Q[pre_x, pre_z, action]));
    }

    void UpdateModelQValue(string prestate, int action, float Reward, string nxtstate)
    {
        int action_index = GetMaxQValue(nxtstate);
        double maxNxtQ = 0;
        if (action_index != -1)
        {
            maxNxtQ = optimizeQ[nxtstate][action_index];
        }
        optimizeQ[prestate][action] += (learningRate * (Reward + discountFactor * maxNxtQ - optimizeQ[prestate][action]));
        //Q[pre_x, pre_z, action] += (learningRate * (Reward + discountFactor * Q[nxt_x, nxt_z, action_index] - Q[pre_x, pre_z, action]));
    }

    void AsyncUpdateQValue(Vector3 state, int action, Vector3 nextState, float Reward)
    {
        //discountFactor = 0.9f; // �ۿ�����
        //learningRate = 0.1f; // ѧϰ��
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

        /*List<int> tmp = new List<int>();
        for (int i = 0; i < numActions; i++)
        {
            if (tmp_Q[x, z, i] == tmp_Q[x, z, action_index])
                tmp.Add(i);
        }

        return tmp[UnityEngine.Random.Range(0, tmp.Count)];*/
        return action_index;
    }

    // ����state״̬�����Qֵ�Ķ�������
    int GetMaxQValue(string state)
    {
        //Dictionary<string, Dictionary<int, double>> tmp_Q;
        /*if (path)
        {
            tmp_Q = qTable.Q_table;
        }
        else
        {
            tmp_Q = optimizeQ;
        }*/
        if (optimizeQ.ContainsKey(state) == false)
        {
            optimizeQ.Add(state, new Dictionary<int, double>());
            return -1;
        }
        // ��ȡ״̬��Ӧ�Ķ��������ֵ�
        var actionRewards = optimizeQ[state];

        
        // ��ʼ�����ֵ�Ͷ�Ӧ�Ķ���
        int maxAction = -1;
        double maxReward = double.MinValue;

        // �������������ֵ䣬�ҵ������ֵ����Ӧ�Ķ���
        foreach (var actionReward in actionRewards)
        {
            if (actionReward.Value > maxReward)
            {
                maxReward = actionReward.Value;
                maxAction = actionReward.Key;
            }
        }

        return maxAction;
    }

    // ��ִ�ж����ڲ��޸�״̬������ֵΪִ�ж������õĽ���
    // ��Agent_idx��״̬stateִ�ж���action
    // ��Agent_idx����������state״̬��ִ�ж���state
    float TakeAction(int Agent_idx, int action)
    {
        // ��state״ִ̬��action����
        int x = (int)Agents[Agent_idx].transform.position.x;
        int z = (int)Agents[Agent_idx].transform.position.z;
        
        Vector3 preState = Agents[Agent_idx].transform.position;
        // �ƶ�֮���x��z����
        x += dx[action];
        z += dz[action];

        // �ƶ�֮��δԽ��
        if (x >= 0 && x < map.width && z >= 0 && z < map.height && !isObstacle(x, z))
        {
            points.Add(new Vector3(x, 4, z));
            // �ƶ�
            float y = GetCurrentStateY(new Vector3(x, 0, z));
            Vector3 next_state = new Vector3(x, y, z);
            if (Agents[Agent_idx] != null)
            {
                Destroy(Agents[Agent_idx]);
            }

            Agents[Agent_idx] = Instantiate(map.agent, next_state, Quaternion.identity);

            // ����״̬
            string pre = Agents_currentState[Agent_idx];
            //Debug.Log("state: " + pre);
            Agents_currentState[Agent_idx] = getState(x, z);
            Agents_previousState[Agent_idx] = pre;

            if (path)
            {
                //Thread.Sleep(3000);
                //Debug.DrawLine(previousState, currentState, Color.red, 100);
            }
            Vector3 currState = Agents[Agent_idx].transform.position;
            // ���ؽ���

            // ��ǰ�����ɹ�������ϰ���
            /*if (action == 4)
            {
                if (isObstacle(x + 1, z) || isObstacle(x, z + 1)) return MyReward(preState, currState, Agent_idx) + 1f;
                else return -1;
            }*/
            float r = MyReward(preState, currState, Agent_idx);
            //float r = PeiImproveReward(preState, currState, Agent_idx);
            //float r = QLearningReward(preState, currState, Agent_idx);
            return r;
        }

        if (isObstacle(x, z)) crash[Agent_idx]++;
        // Խ�緵��-1�Ľ���
        return -1f;
    }

    bool isObstacle(int x, int z)
    {
        if (dynamic == false) return false;
        for (int i = 0; i < map.obstacleNum; i++)
        {
            int ox = (int)map.getObstacles()[i].transform.position.x;
            int oz = (int)map.getObstacles()[i].transform.position.z;
            if (ox == x && oz == z) return true;
        }

        return false;
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


    float MyReward(Vector3 preState, Vector3 currState, int Agent_idx)
    {
        float w = Mathf.Abs(currState.y - preState.y); // ǰ��߶Ȳ�

        value1[Agent_idx] += w + 1;
        value2[Agent_idx] += 1;
        //Debug.Log(w);
        if (IsGoalState(Agents[Agent_idx].transform.position)) return 10f;
        //return -(w + 1);

        int curr_x = (int)currState.x, curr_z = (int)currState.z;
        int goal_x = (int)goalState.x, goal_z = (int)goalState.z;
        float ddx = Math.Abs(curr_x - goal_x), ddz = Math.Abs(curr_z - goal_z);
        //float d = (float)Math.Sqrt(ddx * ddx + ddz * ddz);
        //float d = Math.Max(ddz, ddx); // ���̾���
        ddx += w; ddz += w;
        float d = (float)Math.Sqrt(ddz * ddz + ddx * ddx); // ŷ�Ͼ���
        int width = map.width + 2, height = map.height + 2;
        d = d / (float)(Math.Sqrt(width * width + height * height)); // ��һ��

        float reward = 0.1f * ((float)Math.Pow(e, -2 * d));
        //Debug.Log("reward: " + reward);
        return reward;
    }

    float PeiImproveReward(Vector3 preState, Vector3 currState, int Agent_idx)
    {
        value1[Agent_idx] += Mathf.Abs(currState.y - preState.y) + 1;
        value2[Agent_idx] += 1;
        if (IsGoalState(currState)) return 10f;

        int curr_x = (int)currState.x, curr_z = (int)currState.z;
        int goal_x = (int)goalState.x, goal_z = (int)goalState.z;
        int ddx = Math.Abs(curr_x - goal_x), ddz = Math.Abs(curr_z - goal_z);
        float d = (float)Math.Sqrt(ddx * ddx + ddz * ddz);
        //float d = Math.Max(ddz, ddx); // ���̾���
        d = d / (float)(Math.Sqrt(map.width * map.width * 2));
        float reward = 0.1f * ((float)Math.Pow(e, -2 * d));
        return reward;
    }

    float QLearningReward(Vector3 preState, Vector3 currState, int Agent_idx)
    {
        value1[Agent_idx] += Mathf.Abs(currState.y - preState.y) + 1;
        value2[Agent_idx] += 1;
        if (IsGoalState(currState))
        {
            return 5f; // �����յ�
        }
        return 0f; // û���յ�
    }

    private void WriteToCSV()
    {
        string path = "C:\\Users\\ADMIN\\Desktop\\3D_MAP\\data\\csv\\step\\";//����·��
        string fileName = path + Algorithm + "-" + DateTime.Now.ToString("yyyy-MM-dd-HH") + ".csv";//�ļ���
        string Datedate = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");//������Сʱ������
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        if (!File.Exists(fileName))
        {
            StreamWriter sw = new StreamWriter(fileName, true, Encoding.UTF8);
            string str1 = "episode" + "," + "step" + "\t\n";
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
        string path = "C:\\Users\\ADMIN\\Desktop\\3D_MAP\\data\\csv\\crash\\";//����·��
        string fileName = path + Algorithm + "-Crash-" + DateTime.Now.ToString("yyyy-MM-dd-HH") + ".csv";//�ļ���
        string Datedate = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");//������Сʱ������
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }
        if (!File.Exists(fileName))
        {
            StreamWriter sw = new StreamWriter(fileName, true, Encoding.UTF8);
            string str1 = "episode" + "," + "crash" + "\t\n";
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
        string name = DateTime.Now.ToString("yyyy-MM-dd-hh-mm-s"); // �ļ�����
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
        sw.Flush(); //��ջ�����
        sw.Close(); //�ر���
        fs.Close();
    }

    void OnApplicationQuit()
    {
        foreach (Thread thread in agentThreads)
        {
            if (thread != null && thread.IsAlive)
            {
                thread.Abort();
            }
        }
    }

    public Q_Table ConvertToQTable(Dictionary<string, Dictionary<int, double>> qTableDict)
    {
        Q_Table qTable = new Q_Table();

        foreach (var statePair in qTableDict)
        {
            QState qState = new QState();
            qState.state = statePair.Key;

            foreach (var actionPair in statePair.Value)
            {
                QAction qAction = new QAction();
                qAction.action = actionPair.Key;
                qAction.value = actionPair.Value;

                qState.actions.Add(qAction);
            }

            qTable.states.Add(qState);
        }

        return qTable;
    }
    public void SaveQTableToFile(Dictionary<string, Dictionary<int, double>> qTableDict)
    {
        string name = DateTime.Now.ToString("yyyy-MM-dd-hh-mm-s"); // �ļ�����

        Q_Table qTable = ConvertToQTable(qTableDict);
        string json = JsonUtility.ToJson(qTable, true);
        System.IO.File.WriteAllText("C:\\Users\\ADMIN\\Desktop\\3D_MAP\\data\\qTable\\" + Algorithm + "-" + name, json);
    }

    public Q_Table LoadQTableFromFile(string filePath)
    {
        if (!System.IO.File.Exists(filePath))
        {
            Debug.LogError("File not found");
            return null;
        }

        string json = System.IO.File.ReadAllText(filePath);
        Q_Table qTable = JsonUtility.FromJson<Q_Table>(json);
        return qTable;
    }

    public Dictionary<string, Dictionary<int, double>> ConvertFromQTable(Q_Table qTable)
    {
        Dictionary<string, Dictionary<int, double>> qTableDict = new Dictionary<string, Dictionary<int, double>>();

        foreach (var qState in qTable.states)
        {
            Dictionary<int, double> actionDict = new Dictionary<int, double>();
            foreach (var qAction in qState.actions)
            {
                actionDict[qAction.action] = qAction.value;
            }
            qTableDict[qState.state] = actionDict;
        }

        return qTableDict;
    }
}



