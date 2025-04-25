using UnityEngine;
using System.Collections;
using System.IO;
using System.Text;

public class MeshSaver : MonoBehaviour
{
    public string prefabTag = "FishAgent"; // Tag for the Prefabs
    public string savePath = "FastForward/Meshes"; // Save path
    private int frameCount = 0;
    public Material silverSalmonMaterial; // Assign this in Unity Inspector

    private void Start()
    {
        if (!Directory.Exists(savePath))
        {
            Directory.CreateDirectory(savePath);
        }
    }

    private void Update()
    {
        // Call the method to export meshes every frame
        ExportMeshesForCurrentFrame();
        frameCount++; // Increment frame count for the next frame
    }

    void ExportMeshesForCurrentFrame()
    {
        // Get all Prefabs with the specified tag
        GameObject[] allPrefabs = GameObject.FindGameObjectsWithTag(prefabTag);

        if (allPrefabs == null || allPrefabs.Length == 0)
        {
            Debug.Log("No Prefabs found with the tag: " + prefabTag + ". Retrying...");
            return;
        }

        // File names
        string objFileName = $"AllMeshes_Frame_{frameCount}.obj";
        string mtlFileName = $"AllMeshes_Frame_{frameCount}.mtl";
        string objFilePath = Path.Combine(savePath, objFileName);
        string mtlFilePath = Path.Combine(savePath, mtlFileName);

        StringBuilder objFileContent = new StringBuilder();
        StringBuilder mtlFileContent = new StringBuilder();
        int vertexOffset = 0;

        // Add material to MTL file
        if (silverSalmonMaterial != null)
        {
            AppendMaterialToMtl(silverSalmonMaterial, mtlFileContent);
        }

        // Iterate through all Prefabs
        int meshIndex = 0; // Unique index for each mesh
        foreach (GameObject prefab in allPrefabs)
        {
            // Find the Ginjake_low GameObject
            Transform ginjakeLowTransform = prefab.transform.Find("Ginjake_low");
            if (ginjakeLowTransform != null)
            {
                AppendMeshToObj(ginjakeLowTransform.gameObject, objFileContent, mtlFileContent, ref vertexOffset, meshIndex);
                meshIndex++;
            }
        }

        // Write mesh data to OBJ file
        File.WriteAllText(objFilePath, objFileContent.ToString());
        // Write material data to MTL file
        File.WriteAllText(mtlFilePath, mtlFileContent.ToString());

        Debug.Log("Saved all meshes to: " + objFilePath);
        Debug.Log("Saved materials to: " + mtlFilePath);
    }

    void AppendMeshToObj(GameObject targetFBX, StringBuilder objFileContent, StringBuilder mtlFileContent, ref int vertexOffset, int meshIndex)
    {
        // Get the world-to-local matrix
        Matrix4x4 worldToLocalMatrix = targetFBX.transform.worldToLocalMatrix;

        // Static Mesh
        MeshFilter[] meshFilters = targetFBX.GetComponentsInChildren<MeshFilter>();
        foreach (MeshFilter meshFilter in meshFilters)
        {
            Mesh mesh = meshFilter.mesh;
            if (mesh != null)
            {
                AppendMeshData(mesh, objFileContent, mtlFileContent, ref vertexOffset, targetFBX.transform, meshIndex);
                meshIndex++;
            }
        }

        // Skinned Mesh
        SkinnedMeshRenderer[] skinnedMeshRenderers = targetFBX.GetComponentsInChildren<SkinnedMeshRenderer>();
        foreach (SkinnedMeshRenderer skinnedMeshRenderer in skinnedMeshRenderers)
        {
            Mesh mesh = new Mesh();
            skinnedMeshRenderer.BakeMesh(mesh);
            if (mesh != null)
            {
                AppendMeshData(mesh, objFileContent, mtlFileContent, ref vertexOffset, targetFBX.transform, meshIndex);
                meshIndex++;
            }
        }
    }

    void AppendMeshData(Mesh mesh, StringBuilder objFileContent, StringBuilder mtlFileContent, ref int vertexOffset, Transform parentTransform, int meshIndex)
    {
        // Use predefined material name
        string materialName = "silver_salmon";

        // Create a unique mesh name
        string uniqueMeshName = $"{mesh.name}_{meshIndex}";

        // Add mesh name and material reference
        objFileContent.AppendLine($"g {uniqueMeshName}");
        objFileContent.AppendLine($"usemtl {materialName}");

        // Add vertex coordinates with transformation applied
        foreach (Vector3 v in mesh.vertices)
        {
            Vector3 transformedVertex = parentTransform.TransformPoint(v); // Apply parent transform
            objFileContent.AppendLine($"v {-transformedVertex.x} {transformedVertex.y} {transformedVertex.z}");
        }

        // Add normals with transformation applied (if necessary)
        foreach (Vector3 vn in mesh.normals)
        {
            Vector3 transformedNormal = parentTransform.TransformDirection(vn).normalized; // Apply parent transform
            objFileContent.AppendLine($"vn {-transformedNormal.x} {transformedNormal.y} {transformedNormal.z}");
        }

        // Add UV coordinates
        foreach (Vector2 vt in mesh.uv)
        {
            objFileContent.AppendLine($"vt {vt.x} {vt.y}");
        }

        // Add triangle faces (handle vertex index offset)
        for (int i = 0; i < mesh.subMeshCount; i++)
        {
            int[] triangles = mesh.GetTriangles(i);
            for (int j = 0; j < triangles.Length; j += 3)
            {
                objFileContent.AppendLine($"f {triangles[j + 2] + 1 + vertexOffset}/{triangles[j + 2] + 1 + vertexOffset}/{triangles[j + 2] + 1 + vertexOffset} "
                    + $"{triangles[j + 1] + 1 + vertexOffset}/{triangles[j + 1] + 1 + vertexOffset}/{triangles[j + 1] + 1 + vertexOffset} "
                    + $"{triangles[j] + 1 + vertexOffset}/{triangles[j] + 1 + vertexOffset}/{triangles[j] + 1 + vertexOffset}");
            }
        }

        // Update vertex offset
        vertexOffset += mesh.vertexCount;
    }

    void AppendMaterialToMtl(Material material, StringBuilder mtlFileContent)
    {
        if (material == null)
            return;

        string materialName = "silver_salmon";

        mtlFileContent.AppendLine($"newmtl {materialName}");

        // Base Color
        if (material.HasProperty("_BaseColor"))
        {
            Color baseColor = material.GetColor("_BaseColor");
            mtlFileContent.AppendLine($"Ka {baseColor.r} {baseColor.g} {baseColor.b}"); // Ambient color
            mtlFileContent.AppendLine($"Kd {baseColor.r} {baseColor.g} {baseColor.b}"); // Diffuse color
        }

        // Specular color
        if (material.HasProperty("_SpecColor"))
        {
            Color specularColor = material.GetColor("_SpecColor");
            mtlFileContent.AppendLine($"Ks {specularColor.r} {specularColor.g} {specularColor.b}"); // Specular color
        }

        // Shininess
        if (material.HasProperty("_Smoothness"))
        {
            float shininess = material.GetFloat("_Smoothness") * 100; // Convert smoothness to shininess scale
            mtlFileContent.AppendLine($"Ns {shininess}");
        }

        // Transparency
        if (material.HasProperty("_Opacity"))
        {
            float opacity = material.GetFloat("_Opacity");
            mtlFileContent.AppendLine($"d {opacity}");
        }

        // Emissive color
        if (material.HasProperty("_EmissiveColor"))
        {
            Color emissiveColor = material.GetColor("_EmissiveColor");
            mtlFileContent.AppendLine($"Ke {emissiveColor.r} {emissiveColor.g} {emissiveColor.b}"); // Emissive color
        }

        // Base texture
        if (material.HasProperty("_BaseColorMap"))
        {
            Texture baseTexture = material.GetTexture("_BaseColorMap");
            if (baseTexture != null)
            {
                string texturePath = Path.GetFileNameWithoutExtension(baseTexture.name) + ".png";
                mtlFileContent.AppendLine($"map_Kd {texturePath}"); // Base texture
            }
        }

        // Normal map (if exists)
        if (material.HasProperty("_NormalMap"))
        {
            Texture normalTexture = material.GetTexture("_NormalMap");
            if (normalTexture != null)
            {
                string normalMapPath = Path.GetFileNameWithoutExtension(normalTexture.name) + ".png";
                mtlFileContent.AppendLine($"map_bump {normalMapPath}"); // Normal map
            }
        }
    }
}






