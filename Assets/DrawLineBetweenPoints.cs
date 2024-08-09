using System.Collections.Generic;
using UnityEngine;

public class DrawLineBetweenPoints : MonoBehaviour
{
    public Vector3 previousState; // 设置起点
    public Vector3 currentState; // 设置终点
    public float lineWidth = 10f; // 线条宽度

    private LineRenderer lineRenderer;

    void Start()
    {
        // 获取LineRenderer组件
        lineRenderer = gameObject.GetComponent<LineRenderer>();

        // 检查LineRenderer是否存在
        if (lineRenderer == null)
        {
            Debug.LogError("LineRenderer component is missing. Please add a LineRenderer component to the GameObject.");
            return;
        }

        /*// 配置LineRenderer的材质，如果没有材质的话
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));

        // 设置线条宽度
        lineRenderer.widthMultiplier = lineWidth;*/

        if (lineRenderer != null)
        {
            // 配置LineRenderer的材质
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));

            // 设置线条宽度
            lineRenderer.widthMultiplier = 0.3f;

            // 设置线条的颜色
            /*lineRenderer.startColor = Color.red;
            lineRenderer.endColor = Color.red;*/

            // 设置抗锯齿效果
            lineRenderer.numCapVertices = 10;
            lineRenderer.numCornerVertices = 10;
        }


    }

    public void DrawLineByPoints(List<Vector3> points)
    {
        if (points.Count > 0)
        {
            lineRenderer.positionCount = points.Count;
            lineRenderer.SetPositions(points.ToArray());
        }
    }

    public void DrwaLineByTwoPoints(Vector3 previousState, Vector3 currentState)
    {
        lineRenderer.positionCount = 2;
        lineRenderer.SetPosition(0, previousState);
        lineRenderer.SetPosition(1, currentState);
    }


}
