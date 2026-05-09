using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class TerrainHopperAgent : MarathonAgent {

    TerrainGenerator _terrainGenerator;
    int _lastXPosInMeters;
    int _stepCountAtLastMeter;
    float _pain;
    Vector3 _centerOfMass;

    public override void OnEpisodeBegin()
    {
        base.OnEpisodeBegin();

        BodyParts["pelvis"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="torso");
        BodyParts["foot"] = GetComponentsInChildren<Rigidbody>().FirstOrDefault(x=>x.name=="foot");

        SetCenterOfMass();

        if (_terrainGenerator == null)
            _terrainGenerator = GetComponent<TerrainGenerator>();
        _lastXPosInMeters = (int) BodyParts["foot"].transform.position.x;
        _terrainGenerator.Reset();
        _stepCountAtLastMeter = 0;

        // set to true this to show monitor while training
        //Monitor.SetActive(true);

        StepRewardFunction = StepRewardHopper101;
        TerminateFunction = LocalTerminate;
        ObservationsFunction = ObservationsDefault;
        OnTerminateRewardValue = 0f;
        _pain = 0f;

        base.SetupBodyParts();
        SetCenterOfMass();
    }

    bool LocalTerminate()
    {
        int newXPosInMeters = (int) BodyParts["foot"].transform.position.x;
        if (newXPosInMeters > _lastXPosInMeters) {
            _lastXPosInMeters = newXPosInMeters;
            _stepCountAtLastMeter = this.StepCount;
        }

        SetCenterOfMass();
        var xpos = _centerOfMass.x;
        var terminate = false;
        if (this.StepCount-_stepCountAtLastMeter >= (100*5))
            terminate = true;
        else if (xpos < 2f && _pain > 0f)
            terminate = true;
        else if (_pain > 1f)
            terminate = true;

        return terminate;
    }
    public override void OnTerrainCollision(GameObject other, GameObject terrain) {
        if (terrain.GetComponent<Terrain>() == null)
            return;

        switch (other.name.ToLowerInvariant().Trim())
        {
            // case "torso": // dm_hopper
            //     _pain += 5f;
            //     NonFootHitTerrain = true;
            //     break;
            case "foot": // dm_hopper
            case "calf": // dm_hopper
                FootHitTerrain = true;
                break;
            default:
            case "thigh": // dm_hopper
            case "pelvis": // dm_hopper
            case "torso": // dm_hopper
                _pain += .5f;
                NonFootHitTerrain = true;
                break;
        }
    }

    void ObservationsDefault(VectorSensor sensor)
    {
        var pelvis = BodyParts["pelvis"];
        Vector3 normalizedVelocity = this.GetNormalizedVelocity(pelvis.velocity);
        sensor.AddObservation(normalizedVelocity);
        sensor.AddObservation(pelvis.transform.forward); // gyroscope
        sensor.AddObservation(pelvis.transform.up);

        sensor.AddObservation(SensorIsInTouch);
        foreach (var q in JointRotations) sensor.AddObservation(q);
        sensor.AddObservation(JointVelocity);
        var foot = BodyParts["foot"];
        Vector3 normalizedFootPosition = this.GetNormalizedPosition(foot.transform.position);
        sensor.AddObservation(normalizedFootPosition.y);

        (List<float> distances, float fraction) =
            _terrainGenerator.GetDistances2d(
                pelvis.transform.position, ShowMonitor);

        sensor.AddObservation(distances);
        sensor.AddObservation(fraction);
    }


    void SetCenterOfMass()
    {
        _centerOfMass = Vector3.zero;
        float c = 0f;
        var bodyParts = this.gameObject.GetComponentsInChildren<Rigidbody>();

        foreach (var part in bodyParts)
        {
            _centerOfMass += part.worldCenterOfMass * part.mass;
            c += part.mass;
        }
        _centerOfMass /= c;
    }

    float StepRewardHopper101()
    {
        float uprightBonus = GetDirectionBonus("pelvis", Vector3.forward, 1f);
        uprightBonus = Mathf.Clamp(uprightBonus, 0f, 1f);
        float velocity = Mathf.Clamp(GetNormalizedVelocity("pelvis").x, 0f, 1f);
        // float position = Mathf.Clamp(GetNormalizedPosition("pelvis").x, 0f, 1f);
        float effort = 1f - GetEffortNormalized();

        var reward = velocity;

        _pain = 0f;
        return reward;
    }
}
