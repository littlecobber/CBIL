using Unity.Barracuda;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using MBaske.Sensors.Grid;
//using MBaske.MLUtil;
using System.Threading.Tasks;
using FishSimulation;
using AMP;
using System.Linq;
using System.Net;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using System.IO;

public class FishAgentMulti : Agent
{
    public FishAgentParameters fishAgentParameters;

    private float minSpeed;

    private float maxSpeed;

    private float initialSpeed;

    private float speedThresholdForReward;

    private float deltaVelocityLimit;

    [SerializeField]
    [Tooltip("Reference to sensor component for retrieving detected gameobjects.")]
    private GridSensorComponent3D sensorComponent;
    [SerializeField]
    [Tooltip("Reference to aquarium component that agents stay within.")]
    private Aquarium aquarium;
    public Aquarium Aquarium => aquarium;
    public bool enableGlobalTempAndLight = false;
    public bool enableSocialRankSystem = false;
    [SerializeField]
    [Tooltip("Padding used to restrict agents' initial positions inside the aquarium at start of each episode.")]
    private Vector3 initialStateBoundaryPadding = new Vector3(1f, 1f, 1f);
    //[SerializeField]
    //[Tooltip("Size of roll-back buffer to reset agent's position, rotation and velocity.")]
    private int rollBackBufferSize = 500;                           // 50, 200, 500

    public Vector3 vel { get; protected set; }
    public Vector3 envVel { get; protected set; }
    public string socialRank { get; protected set; }
    public Color defaultColor { get; protected set; }

    [SerializeField]
    private List<NNModel> demoModelAsset;
    [SerializeField]
    private int secondPerModel = 20;
    private string behaviorName;
    private int modelIdx;
    private int modelCount;

    Rigidbody rBody;
    private static string boundaryTag;
    private static string bottomTag;
    private static string chaseTargetTag;
    private static string fishAgentTag;
    private static string fishfoodTag;
    private float detectRange;

    private Quaternion currStepRot;                 // current step rotation
    private float currStepSpeed;                    // current step speed
    private float decisionPeriod;
    private float depthOfFish;

    private Vector3 deltaForward;
    private Vector3 previousPosition;
    private bool collideFlag = false;
    private Queue<Vector3> positionBuffer;
    private Queue<Quaternion> rotationBuffer;
    private Queue<float> speedBuffer;
    private HashSet<GameObject> neighborTargets;
    private List<GameObject> neighborAgents;

    private Vector3 aquaCenter;
    private Vector3 wallExtents;
    private bool isSceneInitialized = false;
    private List<Vector3> aquaVerticesXYZ;

    private float LightIntensityPrefLow;
    private float LightIntensityPrefHigh;
    private float currentSunlightPct;

    private bool chaseModeFlag = false;
    GameObject nearestTargetCandidate;

    //private Color selfStartColor;
    //private Color targetStartColor;
    TrailRenderer trail;
    private float chaseTime = 10.0f;

    private Thread mainThread;
    private bool afterFood = false;
    private GameObject foodObject;

    private float pre_dis = float.NaN;

    // Collect all the fish agents
    private GameObject[] agents;
    private Quaternion[] PreagentRotations;
    private Quaternion[] CurragentRotations;
    private Quaternion StateTransition;

    // Thread for Visual Representation
    public Camera agentCamera;
    private int timestepCounter = 0;
    public int timestepInterval = 10;
    private bool isWaitingForPython = false;
    private int VRcount = 0;
    private static TcpClient client;
    private static NetworkStream stream;
    private static object lockObject = new object(); // threading sync
    private readonly object sendImageLock = new object();
    private int SendImageToPythonCnt = 0;
    private bool CBIL_Flag = false;

    // FishAgentMulti class listener, responsible for updating rewards(Producer)
    private static Thread receiveThread;
    private static List<float> reward_list = new List<float>();
    private static int incremental_cnt = 0;
    private static Mutex init_mutex = new Mutex();
    private static Mutex reward_list_mutex = new Mutex();
    // private TcpClient client;
    // private NetworkStream stream;

    // For high level task
    private Vector3 goal_direction;
    private float goal_distance = 0.0f;
    private float goal_coh_distance = 0.0f;
    private bool feedtrain = false;

    // Reward Stores
    private float high_level_weight = 0.5f;
    private float rule_based_weight = 0.1f;
    private float style_weight = 0.4f;
    private float rule_based_reward = 0.0f; // Since there are many rule based reward, add them up and scale the sum to 0-1.
    private List<float> rule_based = new List<float>();
    private List<float> high_level = new List<float>();
    private List<float> high_level_dot = new List<float>();


    // randomization parameters(Used for more diverse fish motion)
    public float mean = 0.0f;
    public float variance = 1.0f;
    public float rotationNoiseScale = 5.0f;
    public float velocityNoiseScale = 2.5f;

    // responsible for updating rewards on instance
    private Thread UpdateRewardThread;
    private int instance_seq_num;

    // visualization for debug
    // public GameObject arrowPrefab; 
    // private GameObject arrowInstance;
    private List<Vector3> position_info = new List<Vector3>();
    private List<Vector3> velocity_vector = new List<Vector3>();
    private List<Vector3> aquaCenter_info = new List<Vector3>();
    private List<Vector3> Vertical_Vector = new List<Vector3>();

    // Gaussian
    float RandomGaussian(float mean, float variance)
    {
        float u1 = 1.0f - Random.Range(0.0f, 1.0f); //uniform(0,1] random doubles
        float u2 = 1.0f - Random.Range(0.0f, 1.0f);
        float randStdNormal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2); //random normal(0,1)
        return mean + Mathf.Sqrt(variance) * randStdNormal; //random normal(mean,stdDev^2)
    }

    void PauseSimulation()
    {
        Time.timeScale = 0f; // 0 for pausing simulation
    }

    void ContinueSimulation()
    {
        Time.timeScale = 1f; // 1 for continuing simulation
    }

    // send 3D features to python server
    void SendImageToPython()
    {
        lock (sendImageLock)  // safe
        {
            VRcount++;
            // Capture the images from the virtual camera
            RenderTexture renderTexture = agentCamera.targetTexture;
            Texture2D texture2D = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture2D.Apply();

            // convert images to byte
            byte[] imageBytes = texture2D.EncodeToPNG();

            // Clean up the Texture2D after use to free memory
            UnityEngine.Object.Destroy(texture2D);

            string lengthStr = imageBytes.Length.ToString("D10"); // Converts length to a 10-digit string
            byte[] lengthBytes = System.Text.Encoding.UTF8.GetBytes(lengthStr);
            stream.Write(lengthBytes, 0, lengthBytes.Length); // Send the length of the image first
            stream.Write(imageBytes, 0, imageBytes.Length); // Then send the actual image bytes

            if (VRcount == 10)
            {
                VRcount = 0;
                // start to receiving rewards from python
                byte[] buffer = new byte[4]; // assume that the reward is a 4 byte float
                stream.Read(buffer, 0, buffer.Length);

                float probability = System.BitConverter.ToSingle(buffer, 0);

                // compute style reward
                float reward = rewardFunction.ComputeReward(probability);
                // add reward
                AddReward(style_weight * Scale_Style_Reward(reward));

            }
        }  
    }

    // Double check the camera working well
    // (The image is good).
    // System.IO.File.WriteAllBytes(Application.dataPath + "/CapturedImage.png", imageBytes);
    // Debug.Log("Image byte length: " + imageBytes.Length);

    // Send image size as a fixed length header (for example, 10 bytes)
    // Debug.Log("imageBytes "+imageBytes.Length);

    // Start is called before the first frame update
    void Start()
    {
        // This setting is to let simulator has the same time steps
        // for the 30FPS of the reference video
        // It should change depending on the FPS of the reference video
        Time.fixedDeltaTime = 1.0f / 30.0f;

        // Visualization arrow for debugging
        // arrowInstance = Instantiate(arrowPrefab, transform.position, Quaternion.identity);

        // For more diverse training data
        float noiseX = RandomGaussian(mean, variance) * rotationNoiseScale;
        float noiseY = RandomGaussian(mean, variance) * rotationNoiseScale;
        float velocityNoise = RandomGaussian(mean, variance) * velocityNoiseScale;


        /********************************************************************************************************
        // Connect to python server for motion prior
        // If you want to train CBIL without motion prior, you should comment these two lines out
        // ConnectToServer();
        *********************************************************************************************************/


        // AMP Initialization
        discriminatorTrainer = new AMP.AMP.DiscriminatorTrainer();

        // For Food Tracking
        agents = GameObject.FindGameObjectsWithTag("FishAgent");
        foodObject = GameObject.FindWithTag("Food");

        // set up aquarium info
        aquaCenter = Aquarium.center + Aquarium.transform.position;
        wallExtents = new Vector3(Aquarium.center.x, Aquarium.center.y, Aquarium.center.z);
        aquaVerticesXYZ = Aquarium.GetVerticesXYZ();

        // initialize rigdbody
        rBody = GetComponent<Rigidbody>();

        // initialize current (obsolute) rotation and speed
        currStepRot = Quaternion.identity;
        currStepSpeed = 0f;

        // set up speed limits
        var rootAsset = transform.Find("_root1") ? transform.Find("_root1")
                      : transform.Find("madai_bone/_root2") ? transform.Find("madai_bone/_root2")
                      : transform.Find("SeaBream_bone1/_root2") ? transform.Find("SeaBream_bone1/_root2")
                      : transform.Find("salmon_root") ? transform.Find("salmon_root")
                      : transform.Find("Oplegnathus_fasciatus_bone/Oplegnathus_root2") ? transform.Find("Oplegnathus_fasciatus_bone/Oplegnathus_root2")
                      : transform.Find("Yellowfin_SeaBream_bone/_root2") ? transform.Find("Yellowfin_SeaBream_bone/_root2")
                      : transform.Find("root");
        var bodyScale = rootAsset.localScale;
        var fishSize = GetComponentInChildren<SkinnedMeshRenderer>().localBounds.size * transform.localScale.x * bodyScale.x;
        initialSpeed = fishAgentParameters.InitialSpeedInBL * fishSize.x;
        maxSpeed = fishAgentParameters.MaxSpeedInBL * fishSize.x;
        minSpeed = fishAgentParameters.MinSpeedInBL * fishSize.x;
        speedThresholdForReward = fishAgentParameters.SpeedThresholdForRewardInBL * fishSize.x;
        deltaVelocityLimit = fishAgentParameters.DeltaVelocityLimitInBL * fishSize.x * Time.fixedDeltaTime;     // delta velocity limit per simulation step
        //Debug.Log("initialSpeed: " + initialSpeed + ", maxSpeed: " + maxSpeed + ", minSpeed: " + minSpeed + ", speedThresholdForReward: " + speedThresholdForReward);
        //Debug.Log("localBounds.size.x" + GetComponentInChildren<SkinnedMeshRenderer>().localBounds.size.x + ", transform.localScale.x: " + transform.localScale.x + ", fishSize.x: " + fishSize.x);

        // initialize queues and lists
        positionBuffer = new Queue<Vector3>();
        rotationBuffer = new Queue<Quaternion>();
        speedBuffer = new Queue<float>();
        neighborTargets = new HashSet<GameObject>();
        neighborAgents = new List<GameObject>();

        // initialize delta forward direction (to observe)
        deltaForward = new Vector3(0f, 0f, 0f);

        // initialize delta position direction (to observe)
        goal_direction = new Vector3(0f, 0f, 0f); 

        // initialize current depth of fish
        depthOfFish = Mathf.Max((aquaCenter.y * 2 - transform.localPosition.y), 0f);

        // set up tags and detection range (Here can be utilized for chasing behavior tag settings)
        boundaryTag ??= "Boundary";
        bottomTag ??= "Bottom";
        chaseTargetTag ??= "ChaseTarget";
        fishAgentTag ??= "FishAgent";
        fishfoodTag ??= "fishfoodTag";
        detectRange = sensorComponent.MaxDistance;

        // set up light preference
        LightIntensityPrefLow = Random.Range(fishAgentParameters.LowLowerBoundOfLightIntensityPref, fishAgentParameters.HighLowerBoundOfLightIntensityPref);
        LightIntensityPrefHigh = Random.Range(fishAgentParameters.LowUpperBoundOfLightIntensityPref, fishAgentParameters.HighUpperBoundOfLightIntensityPref);
        LightIntensityPrefHigh = LightIntensityPrefHigh > LightIntensityPrefLow ? LightIntensityPrefHigh : (fishAgentParameters.LowUpperBoundOfLightIntensityPref + fishAgentParameters.HighUpperBoundOfLightIntensityPref) / 2;

        decisionPeriod = GetComponent<DecisionRequester>().DecisionPeriod;

        // set up social rank
        var rand = Random.value;
        socialRank = (rand < 0.2) ? "Dominant" : "Subordinate";
        Debug.Log("Social Rank: " + socialRank);
        trail = GetComponent<TrailRenderer>();
        //Debug.Log("trail"+trail);
        defaultColor = GetComponentInChildren<SkinnedMeshRenderer>().material.color;

        // NOTE: for fish size - swim pattern test
        //SetObjectLayer(transform, (int)GameLayers.Outline);

        // NOTE: for checking main thread
        mainThread = System.Threading.Thread.CurrentThread;

        // simulator demo setup
        var bp = GetComponent<BehaviorParameters>();
        behaviorName = bp.BehaviorName;
        modelIdx = 0;
        modelCount = demoModelAsset.Count;
        if (modelCount > 0)
            SetModel(behaviorName, demoModelAsset[modelIdx]);
        //Debug.Log("BehaviorName: " + behaviorName + ", ModelName: " + bp.Model.name);
        if (DemoManager.Instance.sizeDemo) { 
            // For mixed size test demo (NOTE: we should change the materials of fish assets insetead of using this)
            var fishScale = transform.localScale.x;
            if (fishScale == 1.5) GetComponentInChildren<SkinnedMeshRenderer>().material.color = Color.red;
            if (fishScale == 1) GetComponentInChildren<SkinnedMeshRenderer>().material.color = Color.blue;
            if (fishScale == 0.5) GetComponentInChildren<SkinnedMeshRenderer>().material.color = Color.yellow;
        }
        DataGeneratorManager.Instance.click += this.InteractiveRepulsiveForce;
        //Debug.Log("Current temp low pref: " + fishAgentParameters.TempPrefLowCelsius);
        //Debug.Log("Current temp high pref: " + fishAgentParameters.TempPrefHighCelsius);
        //Debug.Log("Current light pref low: " + LightIntensityPrefLow);
        //Debug.Log("Current light pref high: " + LightIntensityPrefHigh);
    }

    // Food Observation
    public Vector3 GetFoodPosition()
    {
        if (foodObject != null)
        {
            return foodObject.transform.position;
        }
        else
        {
            Debug.LogWarning("No game object with tag 'Food' found.");
            return Vector3.zero;
        }
    }


    // Used to scale high level task and rule based reward to the range of (0,1)
    public static float Scale_Reward(float x)
    {
        return 2.0f / (2.0f + Mathf.Exp(x));
    }

    // Used to scale style reward to the range of (0,1)
    public static float Scale_Style_Reward(float x)
    {
        return 2.0f / (1.0f + Mathf.Exp(x));
    }

    // Called at start of each episode (after Start()) to set up environment of new episode
    public override void OnEpisodeBegin()
    {
        // AMP Threading--
        //receiveThread = new Thread(ReceiveData);
        //receiveThread.Start();

        if (neighborAgents == null || neighborTargets == null) this.Start();
        // reset fish position and rotation
        ResetAgentInitialState();

        deltaForward = Vector3.zero;
        goal_direction = Vector3.zero;
        depthOfFish = Mathf.Max((aquaCenter.y * 2 - transform.localPosition.y), 0f);
        neighborTargets.Clear();
        neighborAgents.Clear();
        chaseModeFlag = false;

        isSceneInitialized = true;
    }

    // Helper function to reset the agent's state at start of an episode
    private void ResetAgentInitialState()
    {
        if (!isSceneInitialized)
        {
            //transform.localPosition = aquarium.GetRandomInteriorPointOld(initialStateBoundaryPadding);
            transform.localPosition = aquarium.GetRandomInteriorPoint(initialStateBoundaryPadding.y, initialStateBoundaryPadding.x);
            transform.rotation = Quaternion.Euler(Random.Range(-fishAgentParameters.XRotThresholdInDegree, fishAgentParameters.XRotThresholdInDegree), Random.Range(-180f, 180f), 0f);          //Random.rotationUniform, Quaternion.identity
            currStepSpeed = Random.Range(minSpeed, maxSpeed * 0.2f);
        }
        //else if (positionBuffer.Count > 0 && rotationBuffer.Count > 0 && speedBuffer.Count > 0)
        //{
        //    transform.localPosition = positionBuffer.Peek();
        //    transform.rotation = rotationBuffer.Peek();
        //    currStepSpeed = speedBuffer.Peek();
        //}
        else
        {
            //transform.localPosition = new Vector3(aquaCenter.x, transform.localPosition.y, aquaCenter.z);
            //transform.localPosition = aquaCenter;
            //transform.localPosition = aquarium.GetRandomInteriorPointOld(initialStateBoundaryPadding);
            transform.localPosition = aquarium.GetRandomInteriorPoint(initialStateBoundaryPadding.y, initialStateBoundaryPadding.x);
            transform.rotation = Quaternion.Euler(Random.Range(-fishAgentParameters.XRotThresholdInDegree, fishAgentParameters.XRotThresholdInDegree), Random.Range(-180f, 180f), 0f);          //Random.rotationUniform, Quaternion.identity
            currStepSpeed = Random.Range(minSpeed, maxSpeed * 0.2f);
        }
    }

    // Set up the info to collect by agent (in each cycle)
    public override void CollectObservations(VectorSensor sensor)
    {
        
        neighborAgents.Clear();
        envVel = Vector3.zero;

        // store agents detected by sensor into neighbor list
        foreach (var fish in sensorComponent.GetDetectedGameObjects(fishAgentTag))
        {
            neighborAgents.Add(fish);
        }

        // calculate current sun light intensity percentage based on time
        currentSunlightPct = (Sun.sunlightAngleX < 0) ? 0f : (Sun.sunlightAngleX / Sun.maxSunAngleX);

        // update current depth of fish
        depthOfFish = Mathf.Max((aquaCenter.y * 2 - transform.localPosition.y), 0f);

        /********************************************************************************************************
         * Biological Reward Implementation
        reaction to environment
        ReactToAgents();
        ReactToLight();
        ReactToTemperature();
        if (enableSocialRankSystem)
           ReactToTargets();
        *********************************************************************************************************/


        var obsVecShape = sensor.GetObservationShape();
        // Debug.Log("obsVecShape" + obsVecShape);

        /********************************************************************************************************
        // States needed for different high level task:
        *********************************************************************************************************/

        // Circling:
        // Goal
        sensor.AddObservation(goal_direction);
        sensor.AddObservation(new Vector3(2.5f, 2.5f, 2.5f) - transform.localPosition);
        sensor.AddObservation(Vector3.Distance(transform.localPosition, new Vector3(2.5f, 2.5f, 2.5f)));

        // Fish States
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);


        // Alignment:
        /*
        sensor.AddObservation(deltaforward);
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        */

        /*
        // Aggregation:
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(goal_distance);
        */

        /*
        // Cohesion:
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        sensor.AddObservation(goal_coh_distance);
        */

        // Feeding:
        /*
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        */

        // Chasing:
        /*
        sensor.AddObservation(rBody.velocity);
        sensor.AddObservation(transform.localPosition);
        sensor.AddObservation(transform.forward);
        */
    }

    // Define reactions to neighbor agents (By SoftBank)
    private void ReactToAgents()
    {
        var nearestNeighborForward = new Vector3(0f, 0f, 0f);
        var minDist = float.MaxValue;
        var count = 0;
        foreach (var fish in neighborAgents)
        {
            //Controls the fish avoidence of rocks. Very simplistic. Can sometimes encourage fish to swim into walls
            if (fish.gameObject.name.Contains("Rock_prefab"))
            {
                Vector3 repForce = fish.transform.position - this.transform.position;
                repForce = -repForce.normalized;
                envVel += .025f * repForce;
                SalmonAnimationController fishAni = GetComponent<SalmonAnimationController>();
                if (fishAni != null)
                {
                    // Debug.Log("SSSSSSSSSSSSSSSSSS");
                    fishAni.scatterMotion();
                }
                continue;
            }
            //controls fish attraction to food. Let food handle checks for correct anatanomy
            // if (fish.gameObject.name.Contains("Food"))
            if (foodObject == null)
            {
                Vector3 attForcenew = foodObject.transform.position - fish.transform.position;
                //Vector3 attForcenew = fish.transform.position - transform.position;
                //Vector3 attForce = fish.transform.position - transform.position;
                //Debug.Log("Food position"+foodObject.transform.position);
                envVel += .005f * attForcenew.normalized;  //.125f
                afterFood = true;
                int Timer = 0;
                if (attForcenew.magnitude < 0.01)
                {
                    //Debug.Log("Change food place");
                    //foodObject = null;
                    foodObject.transform.position = Aquarium.center + new Vector3(Random.Range(-1, 1), Random.Range(-0.5f, 0.5f), Random.Range(-1, 1));
                    while (true)
                    {
                        attForcenew = fish.transform.position - transform.position;
                        //attForcenew = foodObject.transform.position - fish.transform.position;
                        envVel += .01f * attForcenew.normalized;
                        Timer++;
                        if (Timer > 5000) { break; }
                    }
                }
                //Debug.Log("Food Found!!!" + attForce);
                continue;
            }
            else
            {
                afterFood = false;
            }
            // reward agent if it stay close to neighbor fish
            var moveDirAlignPct = (90f - Vector3.Angle(fish.transform.forward, transform.forward)) / 90f;
            AddReward(1.5f * moveDirAlignPct);                      //curr: 0.3  [DONO]                                  // 0.1f, 0.2f, 0.5f, 1.0f, 0.8f, 0.7f, 0.9f[DONO], 0.3f, 1.5f, 3f 
            //Debug.Log("fish.forward: " + fish.transform.forward + "; transform.forward: " + 
            //    transform.forward + ", angle" + Vector3.Angle(fish.transform.forward, transform.forward));           

            var dist = Vector3.Distance(transform.localPosition, fish.transform.localPosition);
            if (dist < minDist)
            {
                minDist = dist;
                //nearestNeighborForward = fish.transform.forward;

                // find the nearest target candidate at the front
                var nearestAgent = fish.GetComponent<FishAgentMulti>();
                bool isBehindFlag = IsTargetBehind(fish);
                if (chaseModeFlag == false && socialRank == "Dominant" && nearestAgent.socialRank == "Subordinate" && !isBehindFlag)
                {
                    nearestTargetCandidate = fish;
                }
                //else if (chaseModeFlag == false && socialRank == "Intermediate" && (nearestAgent.socialRank == "Subordinate" || nearestAgent.socialRank == "Intermediate") && !isBehindFlag)
                //{
                //    nearestTargetCandidate = fish;
                //}
                if (!isBehindFlag) 
                    nearestNeighborForward = fish.transform.forward;
            }
        }

        deltaForward = nearestNeighborForward - transform.forward;
        //Debug.Log("nearestNeighborForward: " + nearestNeighborForward + ", transform.forward: " + transform.forward + ", delta: " + deltaForward.magnitude);

        // reward agent if it align with nearest neighbor
        //var moveDirAlignPct = (90f - Vector3.Angle(nearestNeighborForward, transform.forward)) / 90f;
        //AddReward(3.0f * moveDirAlignPct);
    }


    // Define reactions to neighbor targets (By SoftBank)
    private void ReactToTargets()
    {
        // initialize target chasing randomly
        if (chaseModeFlag == false && socialRank == "Dominant" && nearestTargetCandidate && (Random.value < 0.0006f
            || nearestTargetCandidate.GetComponent<FishAgentMulti>().afterFood))               // 0.0001f, 0.0005f, 0.005f
        {
            chaseModeFlag = true;
            neighborTargets.Add(nearestTargetCandidate);

            var targetAgent = nearestTargetCandidate.GetComponent<FishAgentMulti>();
            targetAgent.chaseModeFlag = true;
            //targetAgent.trail.emitting = true;

            GetComponentInChildren<SkinnedMeshRenderer>().material.color = Color.red;
            targetAgent.GetComponentInChildren<SkinnedMeshRenderer>().material.color = Color.blue;

            Debug.Log(socialRank + " starts chasing!!!");
        }

        // chasing behavior
        // NOTE: there's an issue when multiple dominants chase the same target, which only happen ocassionally, needs to be fixed.
        //       Right now after one dominant hits the target, other dominants still have chase mode on but should be turned off (or prohibit shared target?)
        if (chaseModeFlag == true && socialRank == "Dominant")// && neighborTargets.Count > 0)
        {
            // var speed = 5.0f;
            if (chaseTime <= 0.0f)
            {
                //chaseTimeEnd();
                //chaseTime = 100.0f;
            }
            else
            {
                foreach (var target in neighborTargets)
                {
                    var targetAgent = target.GetComponent<FishAgentMulti>();
                    // remove target's chaseModeFlag if it's behind chaser
                    //if (IsTargetBehind(target))
                    if (IsTargetAtSamePosition(target))
                    {
                        targetAgent.chaseModeFlag = false;
                        // targetAgent.trail.emitting = false;
                        targetAgent.GetComponentInChildren<SkinnedMeshRenderer>().material.color = targetAgent.defaultColor;
                        Debug.Log(socialRank + " chase complete...");
                        continue;
                    }

                    // let the predator chase the fish (For video demo test)
                    Vector3 direction = (target.transform.position - transform.position).normalized;
                    envVel += .025f * direction;

                    //transform.position += direction * speed * Time.deltaTime;


                    // reward fish by chasing target
                    Vector3 vecToTarget = target.transform.localPosition - transform.localPosition;
                    float speedToTarget = Vector3.Dot(vecToTarget.normalized, vel);
                    AddReward(8.0f * speedToTarget);            // 1f, 2f

                    // penalize target for being chased               
                    if (speedToTarget > 0)
                    {
                        targetAgent.AddReward(-1.0f * speedToTarget);
                    }

                    // visualization
                    //Debug.DrawLine(transform.localPosition, target.transform.localPosition, Color.magenta, 0.1f, false);
                }

                // remove all targets that are behind the current fish
                neighborTargets.RemoveWhere(IsTargetBehind);

                // no more target in neighborTargets, end chase mode
                /*
                if (neighborTargets.Count == 0)
                {
                    chaseModeFlag = false;

                    GetComponentInChildren<SkinnedMeshRenderer>().material.color = defaultColor;
                    Debug.Log(socialRank + " chase complete...");
                }
                */
            }          
        }
    }

    // Define reaction to light condition:
    //      - bright daylight: fishes form circling pattern, some move to deeper tank, swim faster
    //      - dimmed daylight: fish move to upper tank
    //      - night: fishes stop forming circling behaviour, stay at upper tank, swim slower
    private void ReactToLight()
    {
        // calculate local light intensity around fish (based on Eq.1 on Fore's paper) //Donovan: Disabling light intensity system because it doesnt work for demo purposes  []
        var currentSurfaceLightIntensity =  fishAgentParameters.SurfaceLightIntensityAtNoon* currentSunlightPct;
        if(enableGlobalTempAndLight) currentSurfaceLightIntensity = DataGeneratorManager.Instance.lightIntensity * .017f * currentSunlightPct;
        var localLightIntensity = currentSurfaceLightIntensity * Mathf.Exp(-fishAgentParameters.AttenuationCoeffcient * depthOfFish);

        // if local light intensity isn't within fish's perferred interval, fish will swim to their preferred location (based on Eq.6 of Fore's paper)
        var prefLightIntensityInterval = new Vector2(LightIntensityPrefLow, LightIntensityPrefHigh);
        var deltaVel = Vector3.zero;
        if (localLightIntensity <= prefLightIntensityInterval.x)
        {
            // fish swim to upper tank if local light is below preferred interval
            deltaVel += Vector3.up * ((prefLightIntensityInterval.x - localLightIntensity) / (prefLightIntensityInterval.x - fishAgentParameters.LightResponseSteepnessLow));
            //Debug.Log("Light: Go Up: " + ((prefLightIntensityInterval.x - localLightIntensity) / (prefLightIntensityInterval.x - fishAgentParameters.LightResponseSteepnessLow)));
        }
        else if (localLightIntensity >= prefLightIntensityInterval.y)
        {
            // fish swim to lower tank if local light is above preferred interval
            deltaVel += Vector3.down * ((localLightIntensity - prefLightIntensityInterval.y) / (fishAgentParameters.LightResponseSteepnessHigh - prefLightIntensityInterval.y));
            //Debug.Log("Light: Go Down: " + ((localLightIntensity - prefLightIntensityInterval.y) / (fishAgentParameters.LightResponseSteepnessHigh - prefLightIntensityInterval.y)));
        }
        envVel += deltaVel;
        //Debug.Log("Light: " + deltaVel);
    }


    // Define reaction to local temperature

    private void ReactToTemperature()
    {
        // calculate temperatures of current sea surface (depth < 1m) 
        var seaSurfaceTemp = fishAgentParameters.SeaSurfaceTempCelsius - fishAgentParameters.SeaSurfaceTempDayNightDiffCelsius * (1f - currentSunlightPct);
        if(enableGlobalTempAndLight) seaSurfaceTemp = DataGeneratorManager.Instance.tempMax - fishAgentParameters.SeaSurfaceTempDayNightDiffCelsius * (1f - currentSunlightPct);
        //Debug.Log("Temperture in C: " + seaSurfaceTemp);
        // initialize local sea temp with deeper water (depth >= 1m) temp, which is stable since tank isn't very deep
        var localSeaTemp = seaSurfaceTemp - fishAgentParameters.DaytimeTempIncreaseInShallowWater;

        // if it's daytime and fish is within 1 meter to water surface, its local temp linearly decreases from surface temp to deeper water surface
        if (fishAgentParameters.DayModeFlag && depthOfFish < 1)
        {
            localSeaTemp += fishAgentParameters.DaytimeTempIncreaseInShallowWater * (1f - depthOfFish);
        }

        // if local temp isn't within fish's perferred interval, fish will swim to their preferred location (based on Eq.5 of Fore's paper)
        var deltaVel = Vector3.zero;
        if (localSeaTemp <= fishAgentParameters.TempPrefLowCelsius)
        {
            //fish swim to upper tank if local temp is below preferred interval
            Vector3 dir = Vector3.up;
            deltaVel += dir * ((fishAgentParameters.TempPrefLowCelsius - localSeaTemp) / (fishAgentParameters.TempPrefLowCelsius - fishAgentParameters.TempResponseSteepnessLow));
            //Debug.Log("Temp: Go Up: " + ((fishAgentParameters.TempPrefLowCelsius - localSeaTemp) / (fishAgentParameters.TempPrefLowCelsius - fishAgentParameters.TempResponseSteepnessLow)));
        }
        else if (localSeaTemp >= fishAgentParameters.TempPrefHighCelsius)
        {
            //fish swim to lower tank if local temp is above preferred interval
            Vector3 dir = Vector3.down;
            deltaVel += dir * ((localSeaTemp - fishAgentParameters.TempPrefHighCelsius) / (fishAgentParameters.TempResponseSteepnessHigh - fishAgentParameters.TempPrefHighCelsius));
            //Debug.Log("Temp: Go Down: " + ((localSeaTemp - fishAgentParameters.TempPrefHighCelsius) / (fishAgentParameters.TempResponseSteepnessHigh - fishAgentParameters.TempPrefHighCelsius)));
        }
        envVel += deltaVel;
        //Debug.Log("Temp: " + deltaVel);
    }

    /*********************************************************************************************************
     * Here we define the high level task reward examples
     * Circling
     * Alignment
     * Aggregation
     * Chasing
     * Cohesion
     * Feeding
    *********************************************************************************************************/

    // High Level Tasks Examples:
    // Circling Reward
    private float CalculateCirclingReward(bool clockwise)
    {
        // Calculate the current speed vector of the fish
        // Vector3 fishVel = currStepSpeed * (currStepRot * Vector3.forward);

        // Define the center point around which the fish is circling 
        Vector3 centerPoint = new Vector3(2.5f, 2.5f, 2.5f); ; // aqua center point
        Vector3 fishPosition = transform.localPosition;

        // Calculate the vector from the fish to the center point
        Vector3 toCenter = centerPoint - fishPosition;

        // Calculate the desired tangential direction around the center point
        Vector3 desiredDirection;
        if (clockwise)
        {
            // Clockwise: Calculate the tangential direction using the cross product
            desiredDirection = Vector3.Cross(Vector3.up, toCenter).normalized; // Assuming 'up' is the vertical axis
            goal_direction = desiredDirection;
        }
        else
        {
            // Counterclockwise: Calculate the opposite tangential direction
            desiredDirection = Vector3.Cross(toCenter, Vector3.up).normalized; // Assuming 'up' is the vertical axis
            goal_direction = desiredDirection;
            // Debug.Log("Counterclockwies Direction" + desiredDirection);
        }

        // float targetSpeed = 1.0f; // Target speed range (0.8-1.5)


        // var fishVel = currStepSpeed * (currStepRot * Vector3.forward);
        // var vel_com = fishVel + envVel;

        // Dot target direction
        float dot_vel = Vector3.Dot(desiredDirection, transform.forward);

        // Target velocity
        float target_velocity = 1.0f;
        // Calculate the reward based on how close the fish's direction is to the desired direction
        // float directionReward = Mathf.Exp(-0.25f * (targetSpeed - dot_vel) * (targetSpeed - dot_vel));

        // Calculate the reward based on the speed of the fish
        //float speedReward = Mathf.Exp(-0.25f * Mathf.Pow(targetSpeed - sumVel.magnitude, 2));

        // high_level.Add(high_level_weight * Scale_Reward(10.0f * dot_vel / transform.forward.magnitude)); //for debugging purpose
        // high_level_dot.Add(dot_vel);  // for debugging purpose

        return high_level_weight * Scale_Reward(10.0f *( dot_vel / transform.forward.magnitude) - 10.0f * (rBody.velocity.magnitude - target_velocity)* (rBody.velocity.magnitude - target_velocity));
    }

    // Alignment Reward
    private float CalculateAlignmentReward()
    {
        float reward = 0.0f;
        foreach (var fish in neighborAgents)
        {
            var moveDirAlignPct = (90f - Vector3.Angle(fish.transform.forward, transform.forward)) / 90f;
            reward += moveDirAlignPct;
        }

        return high_level_weight * Scale_Reward(reward);
    }

    // Aggregation Reward
    private float CalculateAggregationReward(float a, float b, float w_agg)
    {
        Vector3 groupCenterPosition = Vector3.zero;
        int N = 0;

        foreach (var fish in agents)
        {
            if (fish != null) 
            {
                Vector3 fishPosition = fish.transform.localPosition;
                groupCenterPosition += fishPosition;
                N++; 
            }
        }

        if (N > 0)
        {
            groupCenterPosition /= N;

            // Calculate the distance between the fish agent and the group center point
            float dist_t = Vector3.Distance(transform.localPosition, groupCenterPosition);
            goal_distance = dist_t;

            // Calculate the reward based on the given formula
            float reward = -w_agg * dist_t / (1.0f + Mathf.Exp(-a * (dist_t - b)));

            return high_level_weight * Scale_Reward(reward);
        }
        else
        {
            Debug.LogWarning("No agents found for aggregation reward calculation.");
            return 0f; 
        }
    }

    // Chasing Reward
    private float CalculateChasingReward(Vector3 dominantPosition, Vector3 subordinatePosition, Vector3 dominantVelocity)
    {
        float chasingWeight = 1.0f;
        // Calculate the position difference vector p_cha*
        Vector3 positionDifference = subordinatePosition - dominantPosition;

        // Normalize the position difference vector
        Vector3 normalizedPositionDifference = positionDifference.normalized;

        // Calculate the reward using the dot product with the dominant fish velocity
        float reward = chasingWeight * Vector3.Dot(normalizedPositionDifference, dominantVelocity);

        return reward;
    }


    // Cohesion Reward
    private float CalculateCohesionReward(float w_coh)
    {
        // Initialize the reward
        float reward = 0.0f;
        int numNeighbors = 0;
        foreach (var fish in neighborAgents)
        {
            reward += Vector3.Distance(transform.localPosition, fish.transform.localPosition);
            numNeighbors++;
        }

        // Average the sum by the number of neighbors
        if (numNeighbors > 0)
        {
            reward /= numNeighbors;
        }

        goal_coh_distance = reward;

        // Multiply by the cohesion weight
        reward *= w_coh;

        return high_level_weight * Scale_Reward(reward);
    }

    // Feeding Reward
    private void CalculateFeeding()
    {
        feedtrain = true;
    }

    /*********************************************************************************************************
     * Receive actions and assign the rewards (in every simulation step)
     * OnActionReceived, you can add your own action update logic
     * Here we use continuous action space
    *********************************************************************************************************/
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        if (DataGeneratorManager.Instance.isAlreadyGeneratingFrame) return;
        var lastStepRot = rBody.rotation;                          // rBody.rotation
        var lastStepSpeed = currStepSpeed;
        MoveAgent(actionBuffers, ref currStepRot, ref currStepSpeed);
        StateTransition = new Quaternion();
        StateTransition = Quaternion.Inverse(lastStepRot) * currStepRot;

        var speedDiff = Mathf.Abs(currStepSpeed - lastStepSpeed);

        // react to boundary objects detected by sensor
        foreach (var boundary in sensorComponent.GetDetectedGameObjects(boundaryTag))
        {            
            var wallDistWeight = 0f;
            // calculate wallDistWeight
            var bPlane = FindBoundaryPlane(boundary);
            var distToPlane = Mathf.Abs(bPlane.GetDistanceToPoint(transform.localPosition)) - aquarium.colliderThickness / 2;
            wallDistWeight = Mathf.Max((detectRange - distToPlane) / detectRange, 0f);

            //Debug.DrawLine(boundary.transform.localPosition, boundary.transform.localPosition + bPlane.normal, Color.magenta, 0.1f, false);
            //Debug.Log("wallDistWeight: " + wallDistWeight + "; detectRange: " + detectRange + "; distToPlane: " + distToPlane
            //    + "; transform.localPosition.x/z: " + transform.localPosition.x + ", " + transform.localPosition.z );
            rule_based_reward += (-2.0f * wallDistWeight); // We accumulate these reward and handle them altogether later
        }

        /*********************************************************************************************************
        //// penalize fish if it is close to bottom
        //foreach (var bottom in sensorComponent.GetDetectedGameObjects(bottomTag))
        //{
        //    var distToTankBottom = transform.localPosition.y - bottom.transform.localPosition.y;
        //    var bottomDistWeight = Mathf.Max((detectRange - distToTankBottom) / detectRange, 0f);
        //    AddReward(-1.0f * bottomDistWeight);
        //    //Debug.Log("distToTankBottom: " + distToTankBottom + "; bottomDistWeight: " + bottomDistWeight + "; detectRange: " + detectRange);
        //}

        //// penalize fish if it is close to water surface 
        //if (depthOfFish <= 0)
        //{
        //    AddReward(-200.0f);                                         // -1f, -10f, -20f, -40f, -80f, -100f, -30f, -50f, -8f, -200f   // -20f for train from scratch
        //    //EndEpisode();
        //}
        //else if (depthOfFish <= detectRange)
        //{
        //    var surfDistWeight = (detectRange - depthOfFish) / detectRange;
        //    AddReward(-1.0f * surfDistWeight);                          // -5f, -3f, -2f
        //}
        *********************************************************************************************************/

        // penalize fish if it is close to vertical boundaries
        var distToTankBottom = Mathf.Max((aquaCenter.y * 2 - depthOfFish), 0f);
        //Debug.Log("distToTankBottom: " + distToTankBottom + "; aquaCenter: " + aquaCenter + "; depthOfFish: " + depthOfFish + "; waterBounds.center: " + Aquarium.WaterBounds.center);
        if (distToTankBottom > 0 && depthOfFish > 0)
        {
            if (depthOfFish <= detectRange)                                 // penalize fish if too close to water surface
            {
                var surfDistWeight = (detectRange - depthOfFish) / detectRange;
                // rule_based_reward += (-5.0f * surfDistWeight);
                AddReward(rule_based_weight * Scale_Reward(- 5.0f * surfDistWeight));                          // -5f, -3f, -2f
                // rule_based.Add(-5.0f * surfDistWeight);
                rule_based_reward = 0.0f;
                EndEpisode();
            }
            if (distToTankBottom <= detectRange)                            // penalize fish if too close to bottom
            {
                var bottomDistWeight = (detectRange - distToTankBottom) / detectRange;
                AddReward(rule_based_weight * Scale_Reward(- 5.0f * bottomDistWeight));                        // -1f, -2f
                rule_based_reward = 0.0f;
                // rule_based_reward += (-5.0f * bottomDistWeight);
                // rule_based.Add(-5.0f * bottomDistWeight);
                EndEpisode();
            }
        }
        else
        {
            AddReward(rule_based_weight * Scale_Reward(- 20.0f));                                         // -1f, -10f, -20f, -40f, -80f, -100f, -30f, -50f, -8f, -200f   // -20f for train from scratch, -300f for TL
            // rule_based_reward += (-20.0f);
            // rule_based.Add(-20.0f);
            rule_based_reward = 0.0f;
            EndEpisode(); //yifan
        }

        /*********************************************************************************************************
        // energy consumption penalty [DISABLED FOR TESTING PURPOSES] 

        // var degreeDiff = Quaternion.Angle(lastStepRot, currStepRot);
        // AddReward(-0.5f * degreeDiff);                                // penalize rotation        // -0.05f, -0.005f, -0.001f, -0.01f, -0.5f
        // AddReward(-10.0f * Mathf.Abs(currStepSpeed - lastStepSpeed));    // penalize change in speed // -0.01f, -0.005f, -0.1f, -0.5f, -2f, -3f, -6f, -12f, -20f

        //var speedDiff = Mathf.Abs(currStepSpeed - lastStepSpeed);
        //var accelP = -30.0f * Mathf.Abs(currStepSpeed - lastStepSpeed);
        //Debug.Log("currStepSpeed: " + currStepSpeed + "; lastStepSpeed: " + lastStepSpeed + "; speedDiff: " + speedDiff + "; accelP: " + accelP);
        *********************************************************************************************************/
        // movement reward 
        if (currStepRot.eulerAngles.x < fishAgentParameters.XRotThresholdInDegree || currStepRot.eulerAngles.x > (360f - fishAgentParameters.XRotThresholdInDegree))    // eulerAngles is always within [0, 360] degrees
        {
            if (currStepSpeed >= speedThresholdForReward)
            {
                // AddReward(rule_based_weight * Scale_Reward(0.5f));                                        // 0.15f, 0.2f, 0.5f, 2f
                rule_based_reward += (2.0f);
                // rule_based.Add(2.0f); // for debugging
            }
        }
        else
        {
            rule_based_reward += (-10.0f);
            // AddReward(rule_based_weight * Scale_Reward(- 10.0f));                                          // penalize excessive depression/elevation angle    // -0.8f, -1.5f, -4f, -8f, -30f, -100f, -50f
            // rule_based.Add(-10.0f); // for debugging
        }


        /*********************************************************************************************************
        // Calculate Rule based Reward:
        *********************************************************************************************************/
        AddReward(rule_based_weight * Scale_Reward(rule_based_reward));
        rule_based_reward = 0.0f; // clear for the next loop

        /*********************************************************************************************************
        // Calculate High Level Task Reward:
        *********************************************************************************************************/

        // Calculate circling reward
        float circlingReward = CalculateCirclingReward(true);  // true for clockwise anf false for counter-clockwise
        AddReward(circlingReward);

        // Calculate alignment reward
        // float alignmentReward = CalculateAlignmentReward();
        // AddReward(alignmentReward);

        // Calculate aggregation reward
        // float aggReward = CalculateAggregationReward(1f, 1f, 5f); // a, b, agg weight
        // AddReward(aggReward);

        // Calculate cohesion reward
        // float cohReward = CalculateCohesionReward(5.0f);
        // AddReward(cohReward);

        // Calculate feeding reward
        // CalculateFeeding();
    
    }

    // Helper function to move agent in each step and return the new rotation
    private void MoveAgent(ActionBuffers actionBuffers, ref Quaternion currStepRot, ref float currStepSpeed)
    {
        var dt = Time.fixedDeltaTime;   // Time.deltaTime;
        var deltaStepSpeed = Mathf.Clamp(actionBuffers.ContinuousActions[0], -1f, 1f) * deltaVelocityLimit;   // calculate delta speed per STEP (scalar), max value is deltaVelocityLimit
        //deltaStepSpeed = Mathf.Clamp(deltaStepSpeed, -deltaVelocityLimit, deltaVelocityLimit);      // clamp delta speed by its limit at each decision period
        currStepSpeed = Mathf.Clamp(currStepSpeed + deltaStepSpeed, minSpeed, maxSpeed);

        var xRotEuler = Mathf.Clamp(actionBuffers.ContinuousActions[1], -1f, 1f);
        var yRotEuler = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);
        var deltaRotQuaternion = Quaternion.Euler(xRotEuler, yRotEuler, 0);// calculate delta rotation per STEP
        var prevStepRot = currStepRot;
        currStepRot = rBody.rotation * deltaRotQuaternion;                                          // rBody.rotation * deltaRotQuaternion

        // Sum up velocity of fish and environmental factors
        //Debug.Log("Before - currStepRot: " + currStepRot + ", currStepRot.x: " + currStepRot.x 
        //    + ", currStepRot.y: " + currStepRot.y + ", currStepSpeed: " + currStepSpeed + ", currStepRot.eulerAngles: " + currStepRot.eulerAngles);
        var fishVel = currStepSpeed * (currStepRot * Vector3.forward);
        var sumVel = fishVel + envVel;

        // set new rotation if elevation angle is not too high, otherwise use old rotation to prevent twirling (USE RotateTowards instead!)
        if (Mathf.Abs(Vector3.Dot(sumVel.normalized, Vector3.up)) < 0.6f) //.95, .925
        {
            //Clamp rotations when sumVel and up are > .99
            //Vector3 f = Vector3.RotateTowards(fishVel.normalized, sumVel.normalized, 1f, 0.05f);
            currStepRot = Quaternion.LookRotation(sumVel, Vector3.up);
            //currStepRot = Quaternion.Euler(f);
            
        }
        /*
        else //If the fish is angled straight up or down, preserve the previous y axis angle (euler angles)
        {
            currStepRot = Quaternion.LookRotation(sumVel, Vector3.up);
            float deltaAngle = currStepRot.eulerAngles.y - prevStepRot.eulerAngles.y;
            deltaAngle = Mathf.Clamp(deltaAngle, -1.5f, 1.5f); //-2 2
            currStepRot.eulerAngles = new Vector3(currStepRot.eulerAngles.x,prevStepRot.eulerAngles.y + deltaAngle, currStepRot.eulerAngles.z);
        }
        */

        currStepSpeed = sumVel.magnitude;
        //Debug.Log("After - currStepRot: " + currStepRot + ", currStepRot.x: " + currStepRot.x 
        //    + ", currStepRot.y: " + currStepRot.y + ", currStepSpeed: " + currStepSpeed + ", currStepRot.eulerAngles: " + currStepRot.eulerAngles);
        var zRotDegree = Mathf.Clamp(currStepRot.eulerAngles.z, -fishAgentParameters.ZRotThresholdInDegree, fishAgentParameters.ZRotThresholdInDegree);  // clamp z-rotation to prevent rolling
        currStepRot.eulerAngles = new Vector3(currStepRot.eulerAngles.x, currStepRot.eulerAngles.y, zRotDegree);

        rBody.MoveRotation(currStepRot);
        vel = currStepSpeed * transform.forward;                                                    // vel stores the velocity at current time step
        rBody.MovePosition(rBody.position + vel * dt);

        CheckOutOfBoundary();

        // Visualization of actionBuffers for debugging



    }
    /*
    // Control the agent using keyboard arrow keys
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // continuousActionsOut[0] = Input.GetAxis("Mouse Y");         // control speed by moving mouse up (increase) and down (decrease)
        //continuousActionsOut[1] = Input.GetAxis("Vertical");
        //continuousActionsOut[2] = Input.GetAxis("Horizontal");
        //continuousActionsOut[1] = -180f;
        //continuousActionsOut[2] = 180f;

        // 
        float speed = 1.0f;
        float rotateUp =  0.0f;
        float rotateDown = 0.0f;

        // 
        continuousActionsOut[0] = speed;
        continuousActionsOut[1] = rotateUp; // 
        continuousActionsOut[2] = rotateDown; // 
    }
    */
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        // 
        float speed = 1.0f;
        continuousActionsOut[0] = speed;

        //
        float rotateSpeed = 1.0f; //
        float rotateDirection = 0.0f; // 
        float rotateSpeed_1 = 1.0f;
        float rotateDirection_1 = 0.0f;

        // 
        if (Input.GetKey(KeyCode.J))
        {
            //
            rotateDirection = -1.0f;
        }
        else if (Input.GetKey(KeyCode.K))
        {
            // 
            rotateDirection = 1.0f;
        }

        if (Input.GetKey(KeyCode.N))
        {
            //
            rotateDirection_1 = -1.0f;
        }
        else if (Input.GetKey(KeyCode.M))
        {
            // 
            rotateDirection_1 = 1.0f;
        }

        // 
        continuousActionsOut[2] = rotateDirection * rotateSpeed;
        continuousActionsOut[1] = rotateDirection_1 * rotateSpeed_1; // 

        if (foodObject != null && Input.GetKey(KeyCode.C))
        {
      
            Quaternion toFoodRotation = foodObject.transform.rotation * Quaternion.Inverse(transform.rotation);
            Vector3 rotationDirection = toFoodRotation.eulerAngles;

            continuousActionsOut[1] = rotationDirection.x * rotateSpeed;
            continuousActionsOut[2] = rotationDirection.y * rotateSpeed;
            //continuousActionsOut[0] = 0.2f;
        }

    }

    // Penalize agent if hit boundary or target
    private void OnCollisionEnter(Collision collision)
    {
        collideFlag = true;

        // penalize agent if hit boundary 
        if (collision.gameObject.CompareTag(boundaryTag) || collision.gameObject.CompareTag(bottomTag))
        {
            AddReward(rule_based_weight*Scale_Reward(-20.0f));                                     // -1f, -0.75f, -0.5f, -10f, -20f, -40f, -80f, -100f, -30f, -50f, -8f   // -20f for train from scratch, -300f for TL
            EndEpisode(); //yifan

            // var bPlane = FindBoundaryPlane(collision.gameObject);           
            // if (Vector3.Dot(bPlane.normal, transform.forward) < 0)
                // transform.localPosition = new Vector3(previousPosition.x, transform.localPosition.y, previousPosition.z);
        }
        // penalize agent if hit chasing target 
        else if (collision.gameObject.CompareTag(chaseTargetTag))
        {
            AddReward(rule_based_weight*Scale_Reward(-0.5f));
            EndEpisode(); //yifan
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        collideFlag = false;
    }

    // Penalize agent if keep hitting boundary or target
    private void OnCollisionStay(Collision collision)
    {
        collideFlag = true;

        // penalize agent if hit boundary 
        if (collision.gameObject.CompareTag(boundaryTag) || collision.gameObject.CompareTag(bottomTag))
        {
            AddReward(rule_based_weight*Scale_Reward(-20.0f));                                     // -1f, -0.75f, -0.5f, -10f, -20f, -40f, -80f, -100f, -30f, -50f, -8f   // -20f for train from scratch, -300f for TL
            EndEpisode(); //yifan

            // var bPlane = FindBoundaryPlane(collision.gameObject);
            // if (Vector3.Dot(bPlane.normal, transform.forward) < 0)
                // transform.localPosition = new Vector3(previousPosition.x, transform.localPosition.y, previousPosition.z);
        }
        // penalize agent if hit chasing target 
        else if (collision.gameObject.CompareTag(chaseTargetTag))
        {
            AddReward(rule_based_weight*Scale_Reward(-0.5f));
            EndEpisode(); //yifan
        }
    }

    // Penalize agent if hit another agent
    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.CompareTag(fishAgentTag))
        {
            // reward if hit targeting agent, penalize if hit other fish
            if (chaseModeFlag == true && socialRank == "Dominant" && Object.ReferenceEquals(other.gameObject, nearestTargetCandidate))
            {
                AddReward(100.0f);               // 8f, 20f

                //end chase mode if touches the target
                neighborTargets.Remove(nearestTargetCandidate);

                //reset chase mode of target too
                var targetAgent = nearestTargetCandidate.GetComponent<FishAgentMulti>();
                targetAgent.chaseModeFlag = false;
                targetAgent.GetComponentInChildren<SkinnedMeshRenderer>().material.color = targetAgent.defaultColor;
            }
            else
            {
                AddReward(rule_based_weight*Scale_Reward(-4.0f));                                       // -1f, -0.6f, -0.3f. -2f, -4f, -6f, 0.3f   // -1f for train from scratch
            }

            // apply repulsive force for colliding fish
            CollideRepulsiveForce(other.gameObject.transform.localPosition, 0.03f);     // 300 fish - 0.05f, 1k fish - 0.03f/0.035f
        }

        // High level task feeding
        if (feedtrain)
        {
            if (other.gameObject.CompareTag(fishfoodTag))
            {
                AddReward(high_level_weight * Scale_Reward(5.0f));
            }
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.gameObject.CompareTag(fishAgentTag))
        {
            // apply repulsive force for colliding fish
            CollideRepulsiveForce(other.gameObject.transform.localPosition, 0.0003f);   // 300/1k fish - 0.0003f 
        }
    }

    // Check if fish go out of bound and stop it if does
    private void CheckOutOfBoundary()
    {
        var minWallSize = aquaCenter - wallExtents;
        var maxWallSize = aquaCenter + wallExtents;

        if (transform.localPosition.y <= minWallSize.y)
            transform.localPosition = new Vector3(transform.localPosition.x, minWallSize.y, transform.localPosition.z);
        if (transform.localPosition.y >= maxWallSize.y)
            transform.localPosition = new Vector3(transform.localPosition.x, maxWallSize.y, transform.localPosition.z);

        //if (transform.localPosition.x >= maxWallSize.x)
        //    transform.localPosition = new Vector3(maxWallSize.x, transform.localPosition.y, transform.localPosition.z);      
        //if (transform.localPosition.z >= maxWallSize.z)
        //    transform.localPosition = new Vector3(transform.localPosition.x, transform.localPosition.y, maxWallSize.z);
        //if (transform.localPosition.x <= minWallSize.x)
        //    transform.localPosition = new Vector3(minWallSize.x, transform.localPosition.y, transform.localPosition.z);
        //if (transform.localPosition.z <= minWallSize.z)
        //    transform.localPosition = new Vector3(transform.localPosition.x, transform.localPosition.y, minWallSize.z);
    }

    // Find a plane where given boundary lies on
    private Plane FindBoundaryPlane(GameObject boundary)
    {
        var boundPose = boundary.transform.localPosition;
        var boundPlane = new Plane(boundary.transform.forward, boundPose);

        return boundPlane;
    }

    // Find a point on boundary plane
    private Vector3 FindPointOnBound(List<Vector3> aquaVertices, Plane boundPlane)
    {
        var closestVertex = Vector3.zero;
        var dist = float.MaxValue;

        // find a cloest vertex to the collider (not on the boundary)
        foreach (var vertex in aquaVerticesXYZ)
        {
            if (boundPlane.GetDistanceToPoint(vertex) < dist)
            {
                dist = boundPlane.GetDistanceToPoint(vertex);
                closestVertex = vertex;
            }
        }

        var resultPlane = new Plane(boundPlane.normal, closestVertex);
        var resultPos = resultPlane.ClosestPointOnPlane(transform.localPosition);
        return resultPos;
    }

    // Check if a target is behind the current fish
    private bool IsTargetBehind(GameObject target)
    {
        var heading = target.transform.localPosition - transform.localPosition;
        var dot = Vector3.Dot(heading, transform.forward);

        bool isBehind = dot < 0 ? true : false;
        return isBehind;
    }

    // Check if a target is got by a predator
    private bool IsTargetAtSamePosition(GameObject target)
    {
        
        var distance = target.transform.localPosition - transform.localPosition;
        bool isAtSamePosition = distance.magnitude == 0;

        return isAtSamePosition;
    }


    public float GetMinSpeed()
    {
        return minSpeed;
    }

    public float GetMaxSpeed()
    {
        return maxSpeed;
    }

    public Vector3 GetVelocity()
    {
        return vel;
    }

    // NOTE: for fish size - swim pattern test, we recursively set layers for all child objects
    void SetObjectLayer(Transform root, int layer)
    {
        root.gameObject.layer = layer;
        foreach (Transform child in root)
            SetObjectLayer(child, layer);
    }

    // NOTE: for checking if on main thread
    bool IsMainThread()
    {
        return mainThread.Equals(System.Threading.Thread.CurrentThread);
    }

    // For showing trajectory in chase mode
    void DrawTrajectory()
    {
        if (trail == null) return;
        //Debug.Log("Drawnig!!!!");
        // disable trail if not in chase mode
        if (chaseModeFlag == false && trail.emitting == true)
            trail.emitting = false;
        //Debug.Log("Drawnig!!!!");
        if (chaseModeFlag == true && trail.emitting == false)
        {
            
            if (socialRank == "Dominant")
            {
                trail.startColor = Color.red;
                trail.endColor = Color.red;
                trail.emitting = true;
            }
            else if (socialRank == "Subordinate")
            {
                trail.startColor = Color.yellow;
                trail.endColor = Color.yellow;
                trail.emitting = true;
            }
        }
    }

    void chaseTimeCountDown()
    {
        if (chaseModeFlag == true && socialRank == "Dominant")
            chaseTime -= Time.fixedDeltaTime;
    }

    void chaseTimeEnd()
    {
        foreach (var target in neighborTargets)
        {
            var targetAgent = target.GetComponent<FishAgentMulti>();
            targetAgent.chaseModeFlag = false;
            targetAgent.GetComponentInChildren<SkinnedMeshRenderer>().material.color = targetAgent.defaultColor;
        }
        neighborTargets.Clear();
        
        chaseModeFlag = false;
        GetComponentInChildren<SkinnedMeshRenderer>().material.color = defaultColor;
        Debug.Log(socialRank + " time up! Chase stop... ");
    }

    /*********************************************************************************************************
     * For debug purpose
     * Save fish agents' 3D trajectories to monitor the training process
     * Also can be used for showing force directions
    *********************************************************************************************************/

    void SaveDataToFile()
    {
        using (StreamWriter writer = new StreamWriter("fish_data.csv", true))
        {
            // for (int i = 0; i < position_info.Count; i++)
            /*
            for (int i = 0; i < rule_based.Count; i++)
            {
                // writer.WriteLine($"{position_info[i].x},{position_info[i].y},{position_info[i].z},{velocity_vector[i].x},{velocity_vector[i].y},{velocity_vector[i].z},{aquaCenter_info[i].x}, {aquaCenter_info[i].y},{aquaCenter_info[i].z},{Vertical_Vector[i].x},{Vertical_Vector[i].y},{Vertical_Vector[i].z} ");
                writer.WriteLine($"{rule_based[i]}");
            }
            */
            for (int i = 0; i < high_level.Count; i++)
            {
                // writer.WriteLine($"{position_info[i].x},{position_info[i].y},{position_info[i].z},{velocity_vector[i].x},{velocity_vector[i].y},{velocity_vector[i].z},{aquaCenter_info[i].x}, {aquaCenter_info[i].y},{aquaCenter_info[i].z},{Vertical_Vector[i].x},{Vertical_Vector[i].y},{Vertical_Vector[i].z} ");
                writer.WriteLine($"{high_level[i]},{high_level_dot[i]}");
            }

        }

        rule_based.Clear();
        high_level.Clear();
        high_level_dot.Clear();
        // position_info.Clear();
        // velocity_vector.Clear();
        // aquaCenter_info.Clear();
        // Vertical_Vector.Clear();
    }


    // FixedUpdate is called once per physical step (default: 50 times/sec)
    void FixedUpdate()
    {
        //var r = base.GetCumulativeReward();
        //Debug.Log("cumulative reward: " + r);

        // save position before simulation update
        if (collideFlag == false)
            previousPosition = rBody.position;

        // display demo info
        float timeNow = Time.fixedTime;     // Time.realtimeSinceStartup;
        if ((modelIdx + 1) < modelCount && timeNow > (secondPerModel * (modelIdx + 1)))
        {
            Debug.Log("Time.fixedTime: " + Time.fixedTime + ", next modelIdx: " + (modelIdx + 1));
            modelIdx += 1;
            SetModel(behaviorName, demoModelAsset[modelIdx]);
        }

        if (enableSocialRankSystem)
        {
            DrawTrajectory();
            chaseTimeCountDown();
        }
        /*********************************************************************************************************
         * CBIL Motion Prior (Style Reward) Implementation
         * If Motion Piror is wanted during the training process, do not comment these lines below
        *********************************************************************************************************/
        // recurrsively dealing with style rewards from motion prior
        // SendImageToPython(); // send images to python

        /* 
        ++SendImageToPythonCnt;
        if (SendImageToPythonCnt < 100 / Time.fixedDeltaTime) return;
        SendImageToPythonCnt = 0;
        SendImageToPython(); // send images to python
        */


        /*********************************************************************************************************
         * Below is for fish agents' 3D trajectory data collection
        *********************************************************************************************************/

        // Vector field visualization
        // var fishVel = currStepSpeed * (currStepRot * Vector3.forward);
        // var sumVel = fishVel + envVel;
        // Debug.Log("Sumvel" + sumVel);
        // Debug.Log("rB-V" + rBody.velocity);

        /*
        position_info.Add(transform.localPosition);
        var fishVel = currStepSpeed * (currStepRot * Vector3.forward);
        var sumVel = fishVel + envVel;
        velocity_vector.Add(sumVel);
        aquaCenter_info.Add(new Vector3(2.5f,2.5f,2.5f));
        Vertical_Vector.Add(Vector3.up);

        if (Time.frameCount % 100 == 0)
        {
            SaveDataToFile();
        }
        */

        /*
        if (!isWaitingForPython)
        {
            timestepCounter++;
            if (timestepCounter >= timestepInterval)
            {
                timestepCounter = 0;
                isWaitingForPython = true;
                PauseSimulation(); // pause the simulation
            }
        }
        */
    }
    public void Unsubscribe()
    {
        DataGeneratorManager.Instance.click -= this.InteractiveRepulsiveForce;
    }
    void InteractiveRepulsiveForce(Vector3 MouseWSPosition)
    {
        //For some reason, the function is still being called even though the class is deleted
        //This is the temporary solution
        if (this == null)
        {
            return;
        }
        
        //Check to see if near mouse
        Vector3 repForce = MouseWSPosition - this.transform.position;
        
        if (repForce.sqrMagnitude > 2)
        {
            return;
        }
        Debug.Log("Fish are moving away");
        SalmonAnimationController fishAni = GetComponent<SalmonAnimationController>();
        if(fishAni != null)
        {
           // Debug.Log("SSSSSSSSSSSSSSSSSS");
            fishAni.scatterMotion();
        }
        repForce = -repForce.normalized;
        envVel += .5f * repForce;
    }

    // Compute repulsive force for colliding agent
    void CollideRepulsiveForce(Vector3 neighborPose, float magnitude)
    {
        //compute repulsive force from neighbor to current fish
        Vector3 repForce = (this.transform.position - neighborPose).normalized;
        envVel += magnitude * repForce;
    }
    

    // Test Code (Ignore this)
    private AMP.AMP.DiscriminatorTrainer discriminatorTrainer;
    private static AMP.AMP.RewardFunction rewardFunction = new AMP.AMP.RewardFunction();

    private void ConnectToServer()
    {
        lock (lockObject) 
        {
            if (client == null) 
            {
                CBIL_Flag = true;
                client = new TcpClient("localhost", 12345);
                stream = client.GetStream();
                // Debug.Log("Connected to server.");
            }
        }
    }

    private void SentandReceiveData(ref Quaternion StateTransition)
    {
        client = new TcpClient();
        client.Connect("localhost", 12345);
        stream = client.GetStream();

        // Send to Server
        Vector3 euler = StateTransition.eulerAngles;
        string message = euler.x + "," + euler.y + "," + euler.z;
        byte[] data = Encoding.ASCII.GetBytes(message);
        stream.Write(data, 0, data.Length);
        Debug.Log("Vectors sent to server.");

        // Received from Server
        byte[] buffer = new byte[1024];
        StringBuilder receivedDataBuilder = new StringBuilder(); // Create for string
        while (true)
        {
            int bytesRead = stream.Read(buffer, 0, buffer.Length);
            if (bytesRead == 0)
            {
                // wait for data
                continue;
            }

            // Add data to StringBuilder
            receivedDataBuilder.Append(Encoding.ASCII.GetString(buffer, 0, bytesRead));

            // Ensure received a complete string
            string receivedData = receivedDataBuilder.ToString().Trim();
            if (!string.IsNullOrEmpty(receivedData))
            {
                float probability;
                if (float.TryParse(receivedData, out probability))
                {
                    // Debug.Log("Probability"+probability);
                    // Compute Reward
                    AddReward(rewardFunction.ComputeReward(probability));
                }
                break; // finish
            }
        }
    }
}
