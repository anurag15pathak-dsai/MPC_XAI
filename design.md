# Design Document: MPC Water Cooling Optimization

## Overview

The MPC Water Cooling Optimization system is a three-layer architecture that combines Model Predictive Control, Explainable AI, and Large Language Models to optimize water usage in data center cooling while maintaining transparency and operator trust.

The system operates in a continuous control loop with three sequential stages:

1. **Control Layer (MPC)**: The MPC Controller reads sensor data, predicts future temperatures using a thermal model, solves a constrained optimization problem to minimize water usage while satisfying temperature constraints, and sends control commands to flow actuators.

2. **Explainability Layer (XAI → LLM)**: 
   - First, the XAI Engine analyzes the MPC optimization solution to extract technical explanations through attribution methods (identifying active constraints, quantifying trade-offs, computing zone influence scores)
   - Then, the LLM Interpreter receives the XAI attribution results and translates them into natural language explanations that operators can understand

3. **Interface Layer**: Provides real-time monitoring, historical analysis, and interactive query capabilities for operators to understand and oversee system behavior.

The flow is: **Sensors → MPC Optimization → XAI Attribution → LLM Translation → Operator Interface**

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Dashboard   │  │  Query API   │  │  Audit Log   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                 Explainability Layer                        │
│  ┌──────────────────────┐    ┌──────────────────────┐     │
│  │    XAI Engine        │───▶│  LLM Interpreter     │     │
│  │  - Constraint        │    │  - Natural Language  │     │
│  │    Analysis          │    │    Generation        │     │
│  │  - Trade-off         │    │  - Context           │     │
│  │    Quantification    │    │    Management        │     │
│  │  - Influence         │    │  - Query Processing  │     │
│  │    Attribution       │    │                      │     │
│  └──────────────────────┘    └──────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Control Layer                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │              MPC Controller                      │      │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │      │
│  │  │  Thermal   │  │ Optimizer  │  │  Actuator  │ │      │
│  │  │   Model    │  │            │  │  Manager   │ │      │
│  │  └────────────┘  └────────────┘  └────────────┘ │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│              Physical Infrastructure                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Temperature  │  │     Flow     │  │   Cooling    │     │
│  │   Sensors    │  │   Actuators  │  │    Zones     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Sensing**: Temperature sensors measure current conditions across all cooling zones
2. **State Estimation**: MPC Controller updates thermal model state based on measurements
3. **Prediction**: Thermal model predicts future temperatures over prediction horizon
4. **Optimization**: Optimizer solves constrained problem to find optimal flow rates
5. **Actuation**: Control commands sent to flow actuators
6. **XAI Attribution**: XAI Engine analyzes the optimization solution to extract technical explanations through attribution methods
7. **LLM Translation**: LLM Interpreter receives XAI attribution results and generates natural language explanations
8. **Presentation**: Natural language explanations displayed to operators via dashboard or query interface

## Components and Interfaces

### MPC Controller

**Responsibilities:**
- Execute control loop at regular intervals (≤5 minutes)
- Read and validate sensor data
- Update thermal model state estimates
- Formulate and solve optimization problem
- Send control commands to actuators
- Log control decisions and performance metrics

**Key Interfaces:**
```
interface MPCController {
  // Main control loop
  executeControlCycle(): ControlDecision
  
  // Sensor integration
  readSensorData(): SensorReadings
  validateSensorData(readings: SensorReadings): ValidationResult
  
  // Optimization
  formulateOptimizationProblem(state: SystemState): OptimizationProblem
  solveOptimization(problem: OptimizationProblem): OptimizationSolution
  
  // Actuation
  sendControlCommands(solution: OptimizationSolution): ActuationResult
  
  // State management
  updateThermalModel(readings: SensorReadings): void
  getSystemState(): SystemState
}
```

### Thermal Model

**Responsibilities:**
- Predict future temperatures based on current state and control actions
- Model heat generation, transfer, and thermal inertia
- Adapt to measured data to maintain accuracy
- Detect and signal when recalibration is needed

**Mathematical Representation:**
The thermal model uses a discrete-time state-space representation:
```
x(k+1) = A*x(k) + B*u(k) + E*d(k)
y(k) = C*x(k)
```
Where:
- x(k): state vector (temperatures at discretized locations)
- u(k): control input vector (flow rates)
- d(k): disturbance vector (heat generation from equipment)
- y(k): measured output vector (sensor temperatures)
- A, B, C, E: system matrices derived from thermal physics

**Key Interfaces:**
```
interface ThermalModel {
  // Prediction
  predictTemperatures(
    currentState: StateVector,
    controlSequence: ControlSequence,
    horizon: number
  ): TemperaturePredictions
  
  // State estimation
  updateState(measurements: SensorReadings): StateVector
  
  // Model adaptation
  computePredictionError(
    predicted: TemperaturePredictions,
    measured: SensorReadings
  ): number
  
  needsRecalibration(): boolean
  recalibrate(historicalData: HistoricalData): void
}
```

### Optimizer

**Responsibilities:**
- Formulate constrained optimization problem
- Solve optimization within time limits
- Validate solution feasibility
- Handle infeasibility through constraint relaxation

**Optimization Problem Formulation:**
```
minimize: Σ(flow_rates) + penalty * Σ(constraint_violations)

subject to:
  T_min ≤ T_predicted(k) ≤ T_max  for all k in horizon
  flow_min ≤ u(k) ≤ flow_max      for all k in control horizon
  |u(k+1) - u(k)| ≤ rate_limit    for all k
  hydraulic_constraints(u(k))      for all k
```

**Key Interfaces:**
```
interface Optimizer {
  // Problem formulation
  formulateProblem(
    model: ThermalModel,
    state: SystemState,
    constraints: Constraints
  ): OptimizationProblem
  
  // Solving
  solve(problem: OptimizationProblem, timeout: number): OptimizationSolution
  
  // Solution validation
  validateSolution(solution: OptimizationSolution): boolean
  
  // Infeasibility handling
  relaxConstraints(problem: OptimizationProblem): OptimizationProblem
}
```

### XAI Engine

**Responsibilities:**
- Analyze optimization solution structure
- Identify active and near-active constraints
- Quantify trade-offs between objectives
- Attribute influence to different zones and factors
- Generate structured technical explanations

**Analysis Methods:**
- **Constraint Analysis**: Identify which constraints are binding (active) or close to binding
- **Sensitivity Analysis**: Compute how objective changes with small constraint perturbations
- **Lagrange Multiplier Analysis**: Extract shadow prices showing constraint importance
- **Influence Attribution**: Determine which zones/sensors most affected the solution

**Key Interfaces:**
```
interface XAIEngine {
  // Main analysis
  analyzeDecision(
    problem: OptimizationProblem,
    solution: OptimizationSolution,
    context: SystemContext
  ): TechnicalExplanation
  
  // Constraint analysis
  identifyActiveConstraints(solution: OptimizationSolution): ConstraintSet
  computeConstraintMargins(solution: OptimizationSolution): ConstraintMargins
  
  // Trade-off analysis
  quantifyTradeoffs(solution: OptimizationSolution): TradeoffMetrics
  
  // Influence attribution
  attributeInfluence(solution: OptimizationSolution): InfluenceScores
  
  // Change detection
  compareDecisions(
    current: ControlDecision,
    previous: ControlDecision
  ): ChangeAnalysis
}
```

### LLM Interpreter

**Responsibilities:**
- Translate technical explanations to natural language
- Process user queries about MPC behavior
- Maintain conversation context for follow-up questions
- Ensure accuracy and consistency in explanations
- Generate clarifying questions for ambiguous queries

**LLM Integration:**
- Use a pre-trained language model (e.g., GPT-4, Claude, or open-source alternative)
- Fine-tune or prompt-engineer for data center domain terminology
- Implement structured prompts that include technical explanation data
- Validate outputs to prevent hallucination or inaccuracy

**Key Interfaces:**
```
interface LLMInterpreter {
  // Explanation generation
  translateExplanation(
    technical: TechnicalExplanation,
    context: SystemContext
  ): NaturalLanguageExplanation
  
  // Query processing
  processQuery(
    query: ExplanationRequest,
    conversationContext: ConversationContext
  ): QueryResponse
  
  // Context management
  updateContext(
    interaction: UserInteraction,
    context: ConversationContext
  ): ConversationContext
  
  // Validation
  validateExplanation(
    explanation: NaturalLanguageExplanation,
    technical: TechnicalExplanation
  ): ValidationResult
}
```

### Actuator Manager

**Responsibilities:**
- Send control commands to physical actuators
- Monitor actuator health and responsiveness
- Detect actuator failures
- Enforce rate limits and safety bounds

**Key Interfaces:**
```
interface ActuatorManager {
  // Command execution
  sendCommands(flowRates: FlowRateVector): ActuationResult
  
  // Health monitoring
  checkActuatorHealth(): ActuatorHealthStatus
  detectFailures(): FailureReport
  
  // Safety enforcement
  enforceRateLimits(
    current: FlowRateVector,
    target: FlowRateVector
  ): FlowRateVector
  
  enforceSafetyBounds(flowRates: FlowRateVector): FlowRateVector
}
```

## Data Models

### Core Data Structures

```typescript
// System state
interface SystemState {
  timestamp: DateTime
  temperatures: TemperatureVector  // Current temperatures at all zones
  flowRates: FlowRateVector        // Current flow rates at all actuators
  thermalState: StateVector        // Internal thermal model state
  disturbances: DisturbanceVector  // Current heat loads
}

// Sensor data
interface SensorReadings {
  timestamp: DateTime
  temperatures: Map<SensorID, Temperature>
  validity: Map<SensorID, boolean>
}

// Control decision
interface ControlDecision {
  timestamp: DateTime
  flowRates: FlowRateVector
  predictedTemperatures: TemperaturePredictions
  objectiveValue: number
  feasible: boolean
}

// Optimization problem
interface OptimizationProblem {
  objectiveFunction: ObjectiveFunction
  constraints: ConstraintSet
  variables: VariableSet
  parameters: ParameterSet
}

// Optimization solution
interface OptimizationSolution {
  optimalControls: ControlSequence
  objectiveValue: number
  lagrangeMultipliers: MultiplierVector
  activeConstraints: ConstraintSet
  solveTime: number
  converged: boolean
}

// Technical explanation
interface TechnicalExplanation {
  decisionID: string
  timestamp: DateTime
  activeConstraints: ConstraintAnalysis[]
  tradeoffs: TradeoffMetrics
  influenceScores: Map<ZoneID, number>
  changeReasons: ChangeAnalysis | null
  optimizationMetrics: OptimizationMetrics
}

// Constraint analysis
interface ConstraintAnalysis {
  constraintType: ConstraintType
  zoneID: ZoneID | null
  margin: number              // Distance to constraint boundary
  lagrangeMultiplier: number  // Shadow price
  active: boolean
}

// Trade-off metrics
interface TradeoffMetrics {
  waterSavings: number          // Percentage vs baseline
  coolingEffectiveness: number  // Percentage of optimal
  constraintUtilization: number // How close to limits
}

// Natural language explanation
interface NaturalLanguageExplanation {
  explanationID: string
  timestamp: DateTime
  summary: string
  detailedExplanation: string[]
  keyFactors: string[]
  recommendations: string[] | null
  sourceExplanationID: string
}

// Explanation request
interface ExplanationRequest {
  requestID: string
  timestamp: DateTime
  userID: string
  query: string
  context: ConversationContext | null
}

// Query response
interface QueryResponse {
  responseID: string
  requestID: string
  timestamp: DateTime
  answer: string
  clarifyingQuestions: string[] | null
  referencedData: DataReference[]
}
```

### Configuration Data

```typescript
interface SystemConfiguration {
  // Control parameters
  controlInterval: Duration
  predictionHorizon: number
  controlHorizon: number
  
  // Physical constraints
  temperatureLimits: Map<ZoneID, TemperatureRange>
  flowRateLimits: Map<ActuatorID, FlowRateRange>
  rateChangeLimits: Map<ActuatorID, number>
  
  // Optimization parameters
  solverTimeout: Duration
  convergenceTolerance: number
  waterMinimizationWeight: number
  constraintPenaltyWeight: number
  
  // Model parameters
  thermalModelMatrices: ThermalModelParameters
  recalibrationThreshold: number
  
  // Explainability parameters
  llmModelName: string
  llmAPIEndpoint: string
  explanationDetailLevel: DetailLevel
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Water Flow Minimization

*For any* system state and optimization problem, the computed optimal solution should have total water flow rate less than or equal to any other feasible solution that achieves the same cooling effectiveness.

**Validates: Requirements 1.1, 1.3**

### Property 2: Water Savings vs Baseline

*For any* scenario with identical initial conditions and disturbances, running the MPC controller should result in lower total water consumption compared to baseline fixed-flow operation, while maintaining all temperatures within safe limits.

**Validates: Requirements 1.2**

### Property 3: Control Cycle Timing

*For any* sequence of control cycles during normal operation, the time interval between consecutive cycles should not exceed 5 minutes.

**Validates: Requirements 1.4**

### Property 4: Temperature Constraint Satisfaction

*For any* optimization solution, all predicted temperatures at all time steps within the prediction horizon should remain within the defined safe limits for their respective cooling zones.

**Validates: Requirements 2.1, 2.3**

### Property 5: Reactive Cooling Response

*For any* control cycle where a temperature sensor reading exceeds the safe upper limit, the next control cycle should compute flow rates for the affected zone that are greater than or equal to the previous cycle's flow rates.

**Validates: Requirements 2.2**

### Property 6: Safety Priority in Infeasibility

*For any* optimization problem that is infeasible with the original water minimization objective, the controller should produce a solution that satisfies all temperature constraints, even if it requires increased water usage.

**Validates: Requirements 2.4**

### Property 7: Thermal Model Prediction Completeness

*For any* temperature prediction computation, the thermal model should incorporate heat generation from equipment (disturbances), thermal inertia (state dynamics), and heat transfer (coupling between zones) in the prediction equations.

**Validates: Requirements 3.1, 3.2, 3.3**

### Property 8: State Estimation Update

*For any* control cycle with valid sensor measurements, the thermal model state should be updated to incorporate the measurement information before computing predictions.

**Validates: Requirements 3.4**

### Property 9: Flow Rate Constraint Satisfaction

*For any* computed control solution, all flow rates should satisfy minimum and maximum bounds for each actuator, and no cooling zone should receive flow below its minimum required threshold.

**Validates: Requirements 4.2, 7.4**

### Property 10: Flow Rate Change Limiting

*For any* pair of consecutive control solutions, the change in flow rate for each actuator should not exceed the defined rate-of-change limit.

**Validates: Requirements 4.4**

### Property 11: Sensor Data Acquisition

*For any* control cycle, all available temperature sensors should be read and their data should be available before the optimization problem is formulated.

**Validates: Requirements 5.1**

### Property 12: Sensor Validation

*For any* sensor reading used in the control algorithm, it should first be validated against physically plausible ranges before being incorporated into state estimation or optimization.

**Validates: Requirements 5.3**

### Property 13: State Estimation Correction

*For any* control cycle where sensor measurements deviate significantly from model predictions, the thermal model state estimate should be adjusted to reduce the prediction error.

**Validates: Requirements 5.4**

### Property 14: Constraint Violation Logging

*For any* constraint violation that occurs during system operation, a log entry should be created containing the timestamp, violation type, affected zone identifier, and violation magnitude.

**Validates: Requirements 6.3**

### Property 15: Multi-Zone Optimization Scope

*For any* optimization problem formulation, all cooling zones in the system should be included as variables or constraints in the problem, ensuring global rather than local optimization.

**Validates: Requirements 7.1**

### Property 16: Hydraulic Coupling Representation

*For any* optimization problem formulation, the constraints should include hydraulic coupling relationships between zones that reflect the physical water distribution network topology.

**Validates: Requirements 7.2**

### Property 17: Coordinated Multi-Zone Control

*For any* scenario where one cooling zone requires increased flow due to elevated temperature, the optimization should adjust flows to other zones in a way that maintains overall water minimization while satisfying all constraints.

**Validates: Requirements 7.3**

### Property 18: Performance Metrics Logging

*For any* one-hour period of operation, the system should compute and log the water savings percentage compared to baseline, and cooling effectiveness metrics for each zone.

**Validates: Requirements 8.1, 8.2**

### Property 19: Historical Data Retention

*For any* data point (temperature, flow rate, optimization objective) recorded during system operation, it should remain accessible in storage for at least 30 days from the time of recording.

**Validates: Requirements 8.4**

### Property 20: Optimization Problem Structure

*For any* optimization problem formulated by the MPC controller, the objective function should be quadratic or linear, and the problem should include explicit constraints for temperature limits, flow limits, and rate-of-change limits.

**Validates: Requirements 9.1**

### Property 21: Solver Performance

*For any* optimization problem of typical size (standard number of zones and horizon length), the solver should return a solution (optimal or feasible) within 30 seconds.

**Validates: Requirements 9.2**

### Property 22: Solution Constraint Validation

*For any* control solution computed by the optimizer, all constraints should be verified as satisfied (within numerical tolerance) before the solution is applied to actuators.

**Validates: Requirements 9.4**

### Property 23: Explanation Generation Completeness

*For any* control decision computed by the MPC controller, the XAI engine should generate a technical explanation that includes: (1) identification of active and near-active constraints, (2) quantification of trade-offs between water minimization and cooling effectiveness, and (3) influence scores for each cooling zone.

**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

### Property 24: Change Explanation

*For any* control decision that differs from the previous decision by more than a defined threshold (e.g., 10% change in any flow rate), the XAI engine should include in the technical explanation an analysis of what factors caused the change.

**Validates: Requirements 10.5**

### Property 25: Natural Language Translation

*For any* technical explanation generated by the XAI engine, the LLM interpreter should produce a corresponding natural language explanation within 5 seconds.

**Validates: Requirements 11.1, 11.4**

### Property 26: Explanation Factor Prioritization

*For any* natural language explanation where multiple factors contributed to the control decision, the factors should be presented in order of decreasing significance (as measured by influence scores or constraint multipliers).

**Validates: Requirements 11.5**

### Property 27: Query Response Generation

*For any* explanation request submitted by a user, the LLM interpreter should generate and return a response that addresses the query content.

**Validates: Requirements 12.1**

### Property 28: Explanation Data Grounding

*For any* natural language explanation that references specific system data (sensor readings, flow rates, temperatures), the referenced values should match the actual values from the corresponding control decision and system state.

**Validates: Requirements 12.4**

### Property 29: Conversation Context Maintenance

*For any* sequence of explanation requests from the same user session, the LLM interpreter should maintain context such that follow-up questions can reference entities and concepts from previous questions without re-specification.

**Validates: Requirements 12.5**

### Property 30: Explanation Accuracy Grounding

*For any* technical explanation generated by the XAI engine, all information in the explanation should be derivable from the actual optimization problem formulation, solution data, and system state at the time of the decision.

**Validates: Requirements 13.1**

### Property 31: Translation Accuracy Preservation

*For any* natural language explanation, the key technical facts (constraint values, flow rates, temperatures, trade-off metrics) should match the corresponding values in the source technical explanation within acceptable rounding.

**Validates: Requirements 13.2**

### Property 32: Explanation Consistency

*For any* two control decisions with identical optimization problems and solutions, the XAI engine should generate technical explanations that are structurally identical (same active constraints, same trade-off values, same influence scores).

**Validates: Requirements 13.3**

### Property 33: No Hallucination in Explanations

*For any* natural language explanation, all factual claims about system behavior, constraint values, or control decisions should be verifiable against the source technical explanation or system logs, with no introduction of unverified information.

**Validates: Requirements 13.4**

### Property 34: Explanation Request Logging

*For any* explanation request submitted to the system, a log entry should be created containing the request timestamp, user identifier, query text, and a unique request ID.

**Validates: Requirements 14.1**

### Property 35: Technical Explanation Retention

*For any* technical explanation generated by the XAI engine, it should remain accessible in storage for at least 30 days from the time of generation.

**Validates: Requirements 14.2**

### Property 36: Explanation Linkage

*For any* natural language explanation stored in the system, it should contain a reference (ID or pointer) to the source technical explanation from which it was generated.

**Validates: Requirements 14.3**

### Property 37: Historical Explanation Retrieval

*For any* request for a historical control decision's explanation, if the decision occurred within the retention period (30 days), the system should retrieve and return the stored explanation for that decision.

**Validates: Requirements 14.4**

## Error Handling

### MPC Controller Error Scenarios

**Sensor Failures:**
- **Detection**: Validate sensor readings against plausible ranges; detect missing data
- **Response**: Use thermal model predictions for failed sensors; log failure
- **Recovery**: Continue operation with reduced sensor set; alert operators

**Actuator Failures:**
- **Detection**: Monitor actuator response; detect non-responsive actuators within one cycle
- **Response**: Recompute optimization excluding failed actuators; increase flow to remaining actuators if needed
- **Recovery**: Attempt actuator reset; alert maintenance

**Optimization Solver Failures:**
- **Non-convergence**: Use previous control solution; log failure; alert operators
- **Infeasibility**: Relax water minimization objective; if still infeasible, activate emergency mode
- **Timeout**: Return best feasible solution found; log timeout; consider reducing problem size

**Model Prediction Errors:**
- **Detection**: Compare predictions to measurements; track error magnitude over time
- **Response**: Trigger recalibration if errors exceed threshold for multiple cycles
- **Recovery**: Update model parameters using system identification; validate improved accuracy

**Emergency Mode:**
- **Trigger**: No feasible solution exists that satisfies temperature constraints
- **Action**: Set all actuators to maximum safe flow rates
- **Exit**: Gradually reduce flows when temperatures stabilize and optimization becomes feasible
- **Logging**: Record all emergency mode activations with triggering conditions

### XAI Engine Error Scenarios

**Optimization Data Unavailable:**
- **Detection**: Check for missing solution data or problem formulation
- **Response**: Generate partial explanation with available data; mark as incomplete
- **Recovery**: Request missing data; regenerate complete explanation if data becomes available

**Constraint Analysis Failures:**
- **Detection**: Numerical issues in computing Lagrange multipliers or sensitivities
- **Response**: Use alternative analysis methods (e.g., constraint margin analysis only)
- **Recovery**: Log numerical issue; continue with reduced explanation detail

**Influence Attribution Errors:**
- **Detection**: Influence scores don't sum to expected total or contain invalid values
- **Response**: Normalize scores or use uniform attribution
- **Recovery**: Investigate root cause; validate computation methods

### LLM Interpreter Error Scenarios

**LLM API Failures:**
- **Detection**: API timeout, connection error, or error response
- **Response**: Retry with exponential backoff (up to 3 attempts)
- **Fallback**: Return structured technical explanation without natural language translation
- **Recovery**: Alert administrators; check API status; consider fallback LLM

**Hallucination Detection:**
- **Detection**: Validate generated explanation against technical explanation facts
- **Response**: Reject explanation; regenerate with stricter prompts
- **Recovery**: Log hallucination instance; adjust prompt engineering

**Query Ambiguity:**
- **Detection**: Low confidence in query interpretation; multiple possible interpretations
- **Response**: Generate clarifying questions for user
- **Recovery**: Process clarified query with additional context

**Context Management Failures:**
- **Detection**: Context size exceeds limits; context retrieval errors
- **Response**: Summarize older context; prioritize recent interactions
- **Recovery**: Maintain essential context; gracefully degrade to stateless responses if needed

### System-Wide Error Scenarios

**Data Storage Failures:**
- **Detection**: Write failures; read failures; storage capacity exceeded
- **Response**: Alert administrators; implement data retention policies
- **Recovery**: Archive old data; expand storage; ensure critical data is preserved

**Performance Degradation:**
- **Detection**: Control cycle time exceeds limits; solver time exceeds limits; explanation generation time exceeds limits
- **Response**: Log performance metrics; alert operators
- **Recovery**: Reduce problem complexity; optimize algorithms; scale resources

**Configuration Errors:**
- **Detection**: Invalid parameter values; inconsistent constraints; missing required configuration
- **Response**: Reject invalid configuration; use default safe values
- **Recovery**: Validate configuration on startup; provide clear error messages

## Testing Strategy

### Dual Testing Approach

The system requires both unit testing and property-based testing for comprehensive validation:

**Unit Tests** focus on:
- Specific examples demonstrating correct behavior
- Edge cases (sensor failures, actuator failures, infeasibility, emergency mode)
- Integration points between components (MPC ↔ XAI, XAI ↔ LLM)
- Error handling paths
- Configuration validation

**Property-Based Tests** focus on:
- Universal properties that hold across all inputs
- Constraint satisfaction across random system states
- Optimization correctness across random scenarios
- Explanation accuracy across random decisions
- Data consistency across random operations

### Property-Based Testing Configuration

**Testing Library**: Use a property-based testing library appropriate for the implementation language:
- Python: Hypothesis
- TypeScript/JavaScript: fast-check
- Java: jqwik
- Other languages: Select appropriate PBT library

**Test Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `Feature: mpc-water-cooling-optimization, Property {number}: {property_text}`

**Example Property Test Structure**:
```python
# Feature: mpc-water-cooling-optimization, Property 4: Temperature Constraint Satisfaction
@given(system_state=system_states(), horizon=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_temperature_constraints_satisfied(system_state, horizon):
    """For any optimization solution, all predicted temperatures should remain within safe limits."""
    controller = MPCController(config)
    solution = controller.solve_optimization(system_state, horizon)
    
    for time_step in range(horizon):
        for zone_id in system_state.zones:
            temp = solution.predicted_temperatures[time_step][zone_id]
            limits = config.temperature_limits[zone_id]
            assert limits.min <= temp <= limits.max
```

### Unit Testing Strategy

**MPC Controller Tests**:
- Test control cycle execution with known sensor data
- Test optimization with simple 2-zone scenarios
- Test sensor failure handling (missing data, invalid data)
- Test actuator failure detection and recovery
- Test emergency mode activation and exit
- Test model recalibration triggering

**Thermal Model Tests**:
- Test prediction accuracy with known dynamics
- Test state estimation with measurement updates
- Test model parameter updates during recalibration
- Test handling of missing measurements

**Optimizer Tests**:
- Test problem formulation correctness
- Test constraint enforcement
- Test infeasibility handling
- Test solver timeout handling

**XAI Engine Tests**:
- Test constraint analysis with known active constraints
- Test trade-off quantification with known solutions
- Test influence attribution with single-zone scenarios
- Test change detection with sequential decisions

**LLM Interpreter Tests**:
- Test translation of sample technical explanations
- Test query processing with example queries
- Test context management with conversation sequences
- Test hallucination detection with invalid explanations
- Test clarifying question generation with ambiguous queries

**Integration Tests**:
- Test end-to-end control cycle (sensors → MPC → actuators → XAI → LLM)
- Test explanation generation for real control decisions
- Test query answering with real system data
- Test data logging and retrieval
- Test system startup and configuration loading

### Test Data Generation

**For Property-Based Tests**:
- Generate random system states with valid temperature and flow ranges
- Generate random thermal model parameters within physical bounds
- Generate random sensor readings with occasional failures
- Generate random optimization problems with varying complexity
- Generate random user queries with different structures

**For Unit Tests**:
- Create fixture data for typical operating scenarios
- Create edge case data (extreme temperatures, minimum flows, maximum flows)
- Create failure scenarios (all sensors failed, all actuators failed)
- Create infeasible scenarios (impossible to satisfy all constraints)

### Coverage Goals

- **Code Coverage**: Minimum 80% line coverage, 70% branch coverage
- **Property Coverage**: All 37 correctness properties must have corresponding property tests
- **Requirement Coverage**: All testable acceptance criteria must be covered by either unit or property tests
- **Error Path Coverage**: All error handling paths must be tested with unit tests

### Continuous Testing

- Run unit tests on every code commit
- Run property tests nightly (due to longer execution time)
- Run integration tests before releases
- Monitor test execution time and optimize slow tests
- Track and report test coverage metrics
