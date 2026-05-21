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

        // set to true this to show monitor while training
        //Monitor.SetActive(true);

        StepRewardFunction = StepRewardAnt101;
        TerminateFunction = TerminateAnt;
        ObservationsFunction = ObservationsDefault;

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x => x.name == "torso_geom");
        SetupBodyParts();
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
