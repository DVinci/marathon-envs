using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class OpenAIAntAgent : MarathonAgent
{
    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        StepRewardFunction = StepRewardAnt101;
        TerminateFunction = TerminateAnt;
        ObservationsFunction = ObservationsDefault;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x => x.name == "torso_geom");
        SetupBodyParts();
        ApplyAnkleInitPose();
    }

    void ApplyAnkleInitPose()
    {
        // init_qpos specifies ankles at ±1.0 rad (57.3°). MarathonSpawner negates ConfigurableJoint
        // axis vs MuJoCo (ToConfigurable line 1230), so sign is inverted:
        // MuJoCo ankle_1 +1.0 rad → Unity -57.3°; ankle_2/3 -1.0 rad → Unity +57.3°
        var ankleInitAngles = new Dictionary<string, float>
        {
            { "ankle_1", -57.3f },
            { "ankle_2",  57.3f },
            { "ankle_3",  57.3f },
            { "ankle_4", -57.3f },
        };
        foreach (var mj in MarathonJoints)
        {
            if (!ankleInitAngles.TryGetValue(mj.JointName, out float angleDeg)) continue;
            var cj = mj.Joint as ConfigurableJoint;
            if (cj == null) continue;
            var worldAxis = cj.transform.TransformDirection(cj.axis);
            cj.transform.rotation = Quaternion.AngleAxis(angleDeg, worldAxis) * cj.transform.rotation;
            var rb = cj.GetComponent<Rigidbody>();
            if (rb != null) { rb.velocity = Vector3.zero; rb.angularVelocity = Vector3.zero; }
        }
    }

    void ObservationsDefault(VectorSensor sensor)
    {
        if (ShowMonitor)
        {
        }

        var pelvis = BodyParts["pelvis"];
        Vector3 normalizedVelocity = GetNormalizedVelocity(pelvis.velocity);
        sensor.AddObservation(normalizedVelocity);
        sensor.AddObservation(pelvis.transform.forward); // gyroscope
        sensor.AddObservation(pelvis.transform.up);

        sensor.AddObservation(SensorIsInTouch);
        foreach (var q in JointRotations) sensor.AddObservation(q);
        sensor.AddObservation(JointVelocity);
        Vector3 normalizedFootPosition = this.GetNormalizedPosition(pelvis.transform.position);
        sensor.AddObservation(normalizedFootPosition.y);

    }

    bool TerminateAnt()
    {
        var pelvis = BodyParts["pelvis"];
		if (pelvis.transform.position.y<0){
			return true;
		}

        var angle = GetForwardBonus("pelvis");
        bool endOnAngle = (angle < .2f);
        return endOnAngle;
    }

    float StepRewardAnt101()
    {
        float velocity = Mathf.Clamp(GetNormalizedVelocity("pelvis").x, -1f, 1f);
        float effort = 1f - GetEffortNormalized();

        velocity *= 0.7f;
        if (velocity >= .3f)
            effort *= 0.3f;
        else
            effort *= Mathf.Max(velocity, 0f);


        var reward = velocity
                     + effort;
        //if (ShowMonitor)
        //{
        //    var hist = new[] {reward, velocity}.ToList();
        //    Monitor.Log("rewardHist", hist.ToArray(), displayType: Monitor.DisplayType.Independent);
        //}

        return reward;
    }
}
