// Smooth Follow from Standard Assets
// If you have C# code and you want to edit SmoothFollow's vars ingame, use this instead.

using UnityEngine;

[AddComponentMenu("Camera-Control/Smooth Follow")]
public class SmoothFollow : MonoBehaviour
{
    public Transform target;
    public float distance = 10.0f;
    public float height = 5.0f;
    public bool clampToFloor;
    public float heightDamping = 2.0f;
    public float rotationDamping = 3.0f;
    // How quickly the camera catches up to the filtered pivot (seconds)
    public float positionSmoothTime = 0.15f;
    // Low-pass filter on the tracked position — filters out steps/bumps (raise to smooth more)
    public float trackingSmoothTime = 0.35f;

    private float _currentRotationAngle;
    private float _currentHeight;
    private Vector3 _positionVelocity;
    private Vector3 _smoothedTargetPos;
    private Vector3 _trackingVelocity;
    private bool _initialized;

    void LateUpdate()
    {
        if (!target) return;

        if (!_initialized)
        {
            _smoothedTargetPos = target.position;
            _currentRotationAngle = target.eulerAngles.y;
            _currentHeight = target.position.y + height;
            _initialized = true;
        }

        // Low-pass filter: removes high-frequency bumps, keeps general trajectory
        _smoothedTargetPos = Vector3.SmoothDamp(_smoothedTargetPos, target.position, ref _trackingVelocity, trackingSmoothTime);

        float wantedRotationAngle = target.eulerAngles.y;
        float wantedHeight = clampToFloor ? height : _smoothedTargetPos.y + height;

        _currentRotationAngle = Mathf.LerpAngle(_currentRotationAngle, wantedRotationAngle, rotationDamping * Time.deltaTime);
        _currentHeight = Mathf.Lerp(_currentHeight, wantedHeight, heightDamping * Time.deltaTime);

        var currentRotation = Quaternion.Euler(0, _currentRotationAngle, 0);
        Vector3 wantedPosition = _smoothedTargetPos - currentRotation * Vector3.forward * distance;
        wantedPosition.y = _currentHeight;

        transform.position = Vector3.SmoothDamp(transform.position, wantedPosition, ref _positionVelocity, positionSmoothTime);
        transform.LookAt(_smoothedTargetPos);
    }
}