using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using MBaske.Sensors.Grid;
//using MBaske.MLUtil;

public class FishAgentSingle : Agent
{
    const float ZRotThresholdInQuaternion = 0.5f;                                           // threshold of rotation about z-axis for penality (in quarter Quaternion.z)

    public float forceMultiplier = 1f;                                                      // a customizable multiplier to control the force magnitude
    public Vector3 aquaCenter = new Vector3(2.5f, 2.5f, 2.5f);
    public Vector3 wallExtents = new Vector3(2.5f, 2.5f, 2.5f);
    public float minimumSpeed = 0.1f;

    public Vector3 vel { get; protected set; }
    //public Vector3 pos { get; protected set; }   
    //public float sideDetectRadius = 0.3f; 
    //public int stepLimit = 5000;

    [SerializeField]
    [Tooltip("Reference to sensor component for retrieving detected gameobjects.")]
    private GridSensorComponent3D sensorComponent;

    Rigidbody rBody;
    private static string boundaryTag;
    private static string chaseTargetTag;
    private Quaternion currStepRot;
    private float currStepSpeed;

    private Vector3 avgNeighborPosition;
    private Vector3 avgNeighborForward;
    private List<GameObject> neighborTargets;   

    // Start is called before the first frame update
    //  OnEpisodeBegin() is called after Start() and seems to have similar function
    void Start()
    {
        // initialize rigdbody
        rBody = GetComponent<Rigidbody>();

        // initialize rotation and speed
        currStepRot = Quaternion.identity;
        currStepSpeed = 0f;

        // initialize list of target positions
        neighborTargets = new List<GameObject>();

        // initialize average neighbor position and rotation (to observe)
        avgNeighborPosition = new Vector3(0f, 0f, 0f);
        avgNeighborForward = new Vector3(0f, 0f, 0f);

        // set up tags
        boundaryTag ??= "Boundary";
        chaseTargetTag ??= "ChaseTarget";
    }

    // Called at start of each episode to set up environment of new episode
    public override void OnEpisodeBegin()
    {
        // reset fish position and rotation
        transform.localPosition = aquaCenter;
        transform.rotation = Quaternion.LookRotation(Vector3.forward, Vector3.up);      // required for reset fish rotation at start of each episode
        rBody.rotation = Quaternion.identity;
        //rBody.velocity = Vector3.forward * minimumSpeed;                              // not used since rBody.velocity seems to be changed only after physics update,
                                                                                        //  so its value doesn't directly reflect change in currStepSpeed at current time step
        vel = Vector3.forward * minimumSpeed;
        avgNeighborPosition = Vector3.zero;
        avgNeighborForward = Vector3.zero;
    }

    // Set up the info to collect by agent
    public override void CollectObservations(VectorSensor sensor)
    {
        // clear the list storing neighbor targets
        neighborTargets.Clear();

        // react to chasing objects detected by sensor
        foreach (var target in sensorComponent.GetDetectedGameObjects(chaseTargetTag))
        {
            neighborTargets.Add(target);
        }
        ReactToNeighbors();

        sensor.AddObservation(transform.localPosition);
        //sensor.AddObservation(rBody.velocity);
        //sensor.AddObservation(rBody.rotation.x);
        //sensor.AddObservation(rBody.rotation.y);
        //sensor.AddObservation(rBody.rotation);
        sensor.AddObservation(currStepRot);
        sensor.AddObservation(vel);
        sensor.AddObservation(avgNeighborPosition);                 
        sensor.AddObservation(avgNeighborForward);
        //sensor.AddObservation(Vector3.Angle(avgNeighborPosition - transform.localPosition, transform.forward));


        // react to boundary objects detected by sensor
        //foreach (var boundary in sensorComponent.GetDetectedGameObjects(boundaryTag))
        //{
        //    //boundPose= boundary.transform.localPosition;
        //    AddReward(-0.005f);                                 // -0.05f
        //}   
    }

    // Helper function to define reactions to neighbor targets
    private void ReactToNeighbors()
    {
        var avgPosition = new Vector3(0f, 0f, 0f);
        var avgForward = new Vector3(0f, 0f, 0f);

        foreach (var target in neighborTargets)
        {
            // reward agent if it stay close to chasing target
            var moveDirAlignPct = (90f - Vector3.Angle(target.transform.forward, transform.forward)) / 90f;
            AddReward(1.0f * moveDirAlignPct);                                                             // 0.1f, 0.2f, 0.5f, 1.0f
            //Debug.Log("target.forward: " + target.transform.forward + "; transform.forward: " + transform.forward + ", angle" + Vector3.Angle(target.transform.forward, transform.forward));

            // find neighbor avg. pos for observation           
            avgPosition += target.transform.localPosition;
            avgForward += target.transform.forward;
        }
        if (neighborTargets.Count != 0)
        {
            avgPosition /= neighborTargets.Count;
            avgForward /= neighborTargets.Count;
        }

        avgNeighborPosition = avgPosition;
        avgNeighborForward = avgForward;
        //Debug.Log("avgNeighborPosition: " + avgNeighborPosition + "; avgNeighborForward: " + avgNeighborForward + ", neighbor.Count" + neighborTargets.Count);
    }

    // Receive actions and assign the rewards
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var lastStepRot = rBody.rotation;
        var lastStepSpeed = currStepSpeed;
       
        MoveAgent(actionBuffers, ref currStepRot, ref currStepSpeed);

        // measure distances
        //var backWallsDistance = transform.localPosition - (aquaCenter - wallExtents);
        //var frontWallsDistance = (aquaCenter + wallExtents) - transform.localPosition;
        //float distToSide1 = backWallsDistance.z;
        //float distToSide2 = frontWallsDistance.z;
        //float distToSide3 = backWallsDistance.x;
        //float distToSide4 = frontWallsDistance.x;

        //Debug.Log("Rot: " + lastStepRot + ", " + currStepRot + ", " + Quaternion.Angle(lastStepRot, currStepRot)
        //    + "; speed: " + currStepSpeed + ", speed compare: " + (currStepSpeed - lastStepSpeed));

        // penalize fish if it reach vertical boundaries and restart episode
        if (transform.localPosition.y < 0 || transform.localPosition.y > (aquaCenter.y + wallExtents.y))
        {
            AddReward(-1.0f);
            EndEpisode();
        }

        // movement reward and energy consumption penalty 
        //Debug.Log("lastStepRot: " + lastStepRot + "; currStepRot: " + currStepRot + "; rBody.rot: " + rBody.rotation + "; Angle: " + Quaternion.Angle(lastStepRot, currStepRot));
        var degreeDiff = Quaternion.Angle(lastStepRot, currStepRot);       
        AddReward(-0.05f * degreeDiff);                                // panalize rotation        // -0.05f, -0.005f, -0.001f, -0.01f
        if (currStepSpeed > 0)                                          // Vector3.Dot(rBody.velocity.normalized, transform.forward) > 0
        {          
            AddReward(-0.005f * (currStepSpeed - lastStepSpeed));       // panalize accelearation   // -0.01f

            // reward fish if it move forward
            if (vel.magnitude >= minimumSpeed)
            {
                if (currStepRot.x < ZRotThresholdInQuaternion && currStepRot.x > -ZRotThresholdInQuaternion)
                    AddReward(0.1f);
                else
                    AddReward(-0.5f);                                   // penalize excessive depression/elevation angle
            }
        }
        else
            AddReward(-0.5f);                                           // penalize backward movement
    }

    // Helper function to move agent in each step and return the new rotation
    private void MoveAgent(ActionBuffers actionBuffers, ref Quaternion currStepRot, ref float currStepSpeed)
    {
        var dt = Time.deltaTime;
        currStepSpeed = actionBuffers.ContinuousActions[0];                              // currStepSpeed is a scalar
        var xRotEuler = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);
        var yRotEuler = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);
        var rotQuaternion = Quaternion.Euler(xRotEuler, yRotEuler, 0);
        currStepRot = rBody.rotation * rotQuaternion;

        rBody.MoveRotation(currStepRot);
        vel = currStepSpeed * transform.forward;                                        // vel stores the velocity at current time step
        rBody.MovePosition(rBody.position + vel * dt * forceMultiplier); 
    }

    // Control the agent using keyboard arrow keys
    public override void Heuristic(in ActionBuffers actionsOut)     
    {        
        var continuousActionsOut = actionsOut.ContinuousActions;
        continuousActionsOut[0] = Input.GetAxis("Mouse Y");         // control speed by moving mouse up (increase) and down (decrease)
        continuousActionsOut[1] = Input.GetAxis("Vertical"); 
        continuousActionsOut[2] = Input.GetAxis("Horizontal");
    }

    // Handle what happen at the start of collision
    private void OnCollisionEnter(Collision collision)
    {
        // if agent hit the side walls, give penalty and restart episode
        if (collision.gameObject.CompareTag(boundaryTag))
        {
            AddReward(-0.5f);                                       // -1f, -0.75f, -0.5f
            EndEpisode();           
        }
        // penalize agent if it hit the chasing target 
        else if (collision.gameObject.CompareTag(chaseTargetTag))
        {
            AddReward(-0.5f);
            EndEpisode();
        }
    }

    //// Handle what happen in the process of collision
    //private void OnCollisionStay(Collision collision)
    //{
    //    // if agent hit the side walls, give penalty and restart episode
    //    if (collision.gameObject.CompareTag("Boundary"))
    //    {
    //        AddReward(-1.0f);
    //        //EndEpisode();      
    //    }
    //
    //    var dist = transform.localPosition - aquaCenter;
    //    if (Mathf.Abs(dist.x) > wallExtents.x || Mathf.Abs(dist.z) > wallExtents.z)
    //        EndEpisode();
    //}

    // Update is called once per frame
    void Update()
    {
        //Debug.DrawRay(transform.localPosition, transform.forward, Color.yellow, 0.01f, false);
    }



}
