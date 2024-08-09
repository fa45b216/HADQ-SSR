using System;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float moveSpeed = 5.0f;
    public float turnSpeed = 3.0f;

    void Update()
    {
        // Camera movement controls
        // Horizontal: A   D
        // Vertical: S    W
        float horizontalInput = Input.GetAxis("Horizontal");
        float verticalInput = Input.GetAxis("Vertical");
        
        Vector3 moveDirection = new Vector3(horizontalInput, 0, verticalInput);
        moveDirection.Normalize();

        transform.Translate(moveDirection * moveSpeed * Time.deltaTime);

        // Camera rotation controls

        float mouseX = Input.GetAxis("Mouse X");
        float mouseY = Input.GetAxis("Mouse Y");

        transform.Rotate(Vector3.up, mouseX * turnSpeed);
        transform.Rotate(-Vector3.right, mouseY * turnSpeed);
    }
}
