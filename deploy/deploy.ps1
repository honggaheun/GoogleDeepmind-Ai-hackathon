param([Parameter(Mandatory)][string]$Project, [Parameter(Mandatory)][string]$Bucket, [string]$Region = "asia-northeast3", [string]$Service = "whereisit")

$ErrorActionPreference = "Continue"

# 1. Deploy service to Cloud Run
$deployOut = gcloud run deploy $Service --source . --project $Project --region $Region --execution-environment gen2 --cpu 2 --memory 4Gi --cpu-boost --min-instances 1 --max-instances 1 --session-affinity --concurrency 30 --timeout 3600 --set-env-vars "EODI_VERTEX_PROJECT=$Project,EODI_VERTEX_LOCATION=global,EODI_OSM_DB=/tmp/korea.sqlite,EODI_OSM_GCS_MOUNT=/gcs" --set-secrets "KAKAO_REST_API_KEY=kakao-rest-api-key:latest,DATA_GO_KR_KEY=data-go-kr-key:latest,MAPS_API_KEY=maps-api-key:latest" --add-volume "name=osm,type=cloud-storage,bucket=$Bucket,readonly=true" --add-volume-mount "volume=osm,mount-path=/gcs" --allow-unauthenticated --quiet 2>&1
$deployOut | ForEach-Object { Write-Host $_ }
$outText = ($deployOut | Out-String)

if ($LASTEXITCODE -ne 0 -or $outText -match "allUsers|Policy|allowedPolicyMemberDomains") {
    if ($outText -match "allUsers|Policy|allowedPolicyMemberDomains|organization") {
        # 2. Retry with --no-invoker-iam-check when organization policy blocks allUsers
        $retryOut = gcloud run deploy $Service --source . --project $Project --region $Region --execution-environment gen2 --cpu 2 --memory 4Gi --cpu-boost --min-instances 1 --max-instances 1 --session-affinity --concurrency 30 --timeout 3600 --set-env-vars "EODI_VERTEX_PROJECT=$Project,EODI_VERTEX_LOCATION=global,EODI_OSM_DB=/tmp/korea.sqlite,EODI_OSM_GCS_MOUNT=/gcs" --set-secrets "KAKAO_REST_API_KEY=kakao-rest-api-key:latest,DATA_GO_KR_KEY=data-go-kr-key:latest,MAPS_API_KEY=maps-api-key:latest" --add-volume "name=osm,type=cloud-storage,bucket=$Bucket,readonly=true" --add-volume-mount "volume=osm,mount-path=/gcs" --no-invoker-iam-check --quiet 2>&1
        $retryOut | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    } else {
        exit $LASTEXITCODE
    }
}

# 3. Print service URL
$serviceUrl = (gcloud run services describe $Service --project $Project --region $Region --format "value(status.url)" | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $serviceUrl) {
    exit 1
}
Write-Output $serviceUrl

# 4. Run smoke test against deployed service
$warmupResp = Invoke-WebRequest -Uri "$serviceUrl/api/warmup" -Method Get -UseBasicParsing -TimeoutSec 60
if ($warmupResp.StatusCode -ne 200) {
    exit 1
}

$payload = '{"text":"지하철 2호선 삼성역 3번 출구 하차 후 직진, 오토웨이 타워"}'
$payloadBytes = [System.Text.Encoding]::UTF8.GetBytes($payload)
$locateResp = Invoke-RestMethod -Uri "$serviceUrl/api/locate_text" -Method Post -ContentType "application/json; charset=utf-8" -Body $payloadBytes -TimeoutSec 30
$sessionId = $locateResp.session_id
if (-not $sessionId) {
    exit 1
}

$deadline = (Get-Date).AddSeconds(150)
$session = $null
while ((Get-Date) -lt $deadline) {
    $session = Invoke-RestMethod -Uri "$serviceUrl/api/session/$sessionId" -Method Get -TimeoutSec 15
    if ($session.done) {
        break
    }
    Start-Sleep -Seconds 1
}

if (-not $session -or -not $session.done) {
    exit 1
}

$decisionKind = $session.decision.kind
$candidateId = $session.decision.candidate_id
$candidateName = $candidateId
foreach ($c in $session.candidates) {
    if ($c.id -eq $candidateId -and $c.name) {
        $candidateName = $c.name
        break
    }
}
Write-Output "$decisionKind $candidateName"
