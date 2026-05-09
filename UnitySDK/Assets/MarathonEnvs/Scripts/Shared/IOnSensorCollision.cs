using UnityEngine;

public interface IOnSensorCollision
{
     void OnSensorCollisionEnter(Collider sensorCollider, GameObject other);
     void OnSensorCollisionExit(Collider sensorCollider, GameObject other);

}