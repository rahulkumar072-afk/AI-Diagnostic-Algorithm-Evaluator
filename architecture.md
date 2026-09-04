# Proposed System Architecture

## Architecture Layers

### 1. Data Layer
- Electronic health record / diagnostic data source
- Authorised data access
- Minimum necessary data fields

### 2. Validation Layer
- Data type checks
- Range checks
- Missing-data checks
- Model-population eligibility checks

### 3. AI Layer
- Validated machine-learning model
- Version-controlled preprocessing
- Risk / decision-support output

### 4. Explanation Layer
- Feature contribution or interpretable summary
- Confidence/uncertainty information where appropriate
- Clear limitation message

### 5. Application Layer
- Secure clinician dashboard
- Result display
- Human override
- Out-of-scope warning

### 6. Integration Layer
- API-based integration
- HL7/FHIR may be considered where supported by the institution
- Secure data exchange

### 7. Security & Monitoring
- Authentication
- Role-based access
- Encryption
- Audit logging
- Uptime monitoring
- Model/data drift monitoring
- Incident response

## Fail-Safe Design

If the AI service is unavailable, the normal clinical workflow must continue without being blocked by the AI system.
