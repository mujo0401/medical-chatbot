# test_service_principal.py
import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Testing Service Principal Authentication")
print("=" * 60)

# Test 1: Check Environment Variables
print("1. Service Principal Environment Variables:")
sp_vars = {
    'AZURE_CLIENT_ID': os.getenv('AZURE_CLIENT_ID'),
    'AZURE_CLIENT_SECRET': os.getenv('AZURE_CLIENT_SECRET'),
    'AZURE_TENANT_ID': os.getenv('AZURE_TENANT_ID'),
    'AZURE_SUBSCRIPTION_ID': os.getenv('AZURE_SUBSCRIPTION_ID'),
    'AZURE_RESOURCE_GROUP': os.getenv('AZURE_RESOURCE_GROUP'),
    'AZURE_WORKSPACE_NAME': os.getenv('AZURE_WORKSPACE_NAME')
}

for key, value in sp_vars.items():
    if value:
        if 'SECRET' in key:
            display = value[:8] + "*" * (len(value) - 8)
        else:
            display = value
        print(f"   ✅ {key}: {display}")
    else:
        print(f"   ❌ {key}: NOT SET")

# Test 2: Test Environment Credential
print("\n2. Testing Environment Credential:")
try:
    from azure.identity import EnvironmentCredential
    
    print("   🔧 Creating EnvironmentCredential...")
    env_cred = EnvironmentCredential()
    
    print("   🎯 Testing token acquisition...")
    token = env_cred.get_token("https://management.azure.com/.default")
    print("   ✅ EnvironmentCredential works!")
    print(f"   ⏰ Token expires: {token.expires_on}")
    
    # Test 3: Test Azure ML Client with Service Principal
    print("\n3. Testing Azure ML Client with Service Principal:")
    from azure.ai.ml import MLClient
    
    print("   🔧 Creating MLClient with EnvironmentCredential...")
    ml_client = MLClient(
        credential=env_cred,
        subscription_id=sp_vars['AZURE_SUBSCRIPTION_ID'],
        resource_group_name=sp_vars['AZURE_RESOURCE_GROUP'],
        workspace_name=sp_vars['AZURE_WORKSPACE_NAME']
    )
    
    print("   ✅ MLClient created successfully!")
    
    # Test 4: Connect to Workspace
    print("\n4. Testing Workspace Connection:")
    try:
        workspace = ml_client.workspaces.get(sp_vars['AZURE_WORKSPACE_NAME'])
        print(f"   ✅ Connected to workspace: {workspace.name}")
        print(f"   📍 Location: {workspace.location}")
        print(f"   🏷️  Description: {workspace.description or 'Azure ML Workspace'}")
        print(f"   💰 SKU: {workspace.sku.tier}")
        
        # Test 5: List Compute Targets
        print("\n5. Testing Compute Resources:")
        try:
            compute_targets = list(ml_client.compute.list())
            if compute_targets:
                print(f"   ✅ Found {len(compute_targets)} compute targets:")
                for compute in compute_targets:
                    print(f"      - {compute.name} ({compute.type}) - State: {compute.state}")
            else:
                print("   ⚠️  No compute targets found (this is normal for new workspaces)")
                print("   💡 You can create compute targets through Azure ML Studio")
        except Exception as compute_error:
            print(f"   ⚠️  Could not list compute targets: {compute_error}")
        
        # Test 6: Test Experiments/Jobs Access
        print("\n6. Testing Jobs Access:")
        try:
            jobs = list(ml_client.jobs.list(max_results=5))
            print(f"   ✅ Jobs access working! Found {len(jobs)} recent jobs")
        except Exception as jobs_error:
            print(f"   ⚠️  Jobs access issue: {jobs_error}")
            
    except Exception as ws_error:
        print(f"   ❌ Workspace connection failed: {ws_error}")
        
        if "not found" in str(ws_error).lower():
            print(f"   💡 Workspace '{sp_vars['AZURE_WORKSPACE_NAME']}' not found")
            print("   🔍 Available workspaces from resource list:")
            print("      - equanoxai-ws (westus2)")
            print("      - central (centralus)")
            print("      - equanoxai-resource-NorthCentral (northcentralus)")
            print("      - equanoxai-resource-southcentral (westus2)")

except ImportError as import_error:
    print(f"   ❌ Azure SDK import failed: {import_error}")
except Exception as e:
    print(f"   ❌ Environment credential test failed: {e}")

print("\n" + "=" * 60)
print("🎯 Summary:")
print("If all tests pass: Your Azure ML is fully working!")
print("If workspace connection fails: Check workspace name in .env")
print("If authentication works: Your medical chatbot is ready for Azure ML!")

print("\n🚀 Next Steps:")
print("1. Fix CommandJob imports in medical_trainer.py") 
print("2. Run: python app.py")
print("3. Your medical chatbot should have Azure ML capabilities!")