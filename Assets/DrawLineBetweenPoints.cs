using System.Collections.Generic;
using UnityEngine;

public class DrawLineBetweenPoints : MonoBehaviour
{
    public Vector3 previousState; // �������
    public Vector3 currentState; // �����յ�
    public float lineWidth = 10f; // �������

    private LineRenderer lineRenderer;

    void Start()
    {
        // ��ȡLineRenderer���
        lineRenderer = gameObject.GetComponent<LineRenderer>();

        // ���LineRenderer�Ƿ����
        if (lineRenderer == null)
        {
            Debug.LogError("LineRenderer component is missing. Please add a LineRenderer component to the GameObject.");
            return;
        }

        /*// ����LineRenderer�Ĳ��ʣ����û�в��ʵĻ�
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));

        // �����������
        lineRenderer.widthMultiplier = lineWidth;*/

        if (lineRenderer != null)
        {
            // ����LineRenderer�Ĳ���
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));

            // �����������
            lineRenderer.widthMultiplier = 0.3f;

            // ������������ɫ
            /*lineRenderer.startColor = Color.red;
            lineRenderer.endColor = Color.red;*/

            // ���ÿ����Ч��
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
