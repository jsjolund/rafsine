require "helpers"
require "operators"
require "NodeDescriptor"
require "ThreadLayout"
require "KernelGenerator"
require "MemoryManager"
require "MainGenerator"
require "StatisticsGenerator"
require "OpenGLRenderGenerator"
require "InitialisationMethods"
require "WaitGenerator"
require "SavingGenerator"
require "SmagorinskyDynamics"
require "MRTDellar2002"
require "MRTLallemand2000"
require "MRTDHumieresD3Q15_2002"
require "DocumentationGenerator"
require "TRT"
require "EarlyExitGenerator"
require "CascadedLBM_D2Q9"
--require "CascadedLBM_D2Q9_MRTform"
require "CascadedLBM_D3Q27"

--TODO filesystem class to handle folders
src_destination_folder = "./generated_code"
-- folder for documentation
doc_destination_folder = "./generated_documentation"

node = D3Q19Descriptor
--dynamics = SinglePhaseDyn
--dynamics = dynTRT
dynamics = SmagorinskyDyn
--dynamics = dynDellar2002
--dynamics = dynLallemand2000
--dynamics = dynDHumieresD3Q15
--dynamics = dynCLBM_D2Q9
--dynamics = dynCLBM_D3Q27
Re = 30000
Ma = 0.07
N = 128
nu = Ma * N / Re
tau = 3 * nu + 1 / 2
--compute TRT relaxations
lambdaEO = 1 / 4 -- (magic parameter)
lambdaE = -1 / tau
--lambdaO = 1/( (lambdaEO/( 1/2 + 1/lambdaE )) - 1/2 )
lambdaO = -1 / tau
--model  = BGKModel("single_precision", { tau = tau, size = {N,N,N} } )
--model  = TRT("single_precision", { lambdaE = lambdaE, lambdaO = lambdaO, size = {N,N,N} } )
model = SmagorinskyModel("double_precision", {nu = nu, C = 0.02, size = {N, N, N}})
--model  = MRTDellar2002("single_precision", { tau = tau, tauJ = 0.7, tauN = 0.7, size = {N,N} } )
--model  = MRTLallemand2000("single_precision", { s2 = 1/tau, s8 = 1/tau, s3 = 1/tau - 0.2, s5 = 1/tau, s7 = 1/tau, alpha3 = 4, gamma4 = -18, size = {N,N} } )
--model  = MRTDHumieresD3Q15_2002("double_precision", { s1 = 1/tau, s2 = 1/tau, s4 = 1/tau, s9 = 1/tau, s14 = 1/tau, We = 1, Wej = -5, size = {N,N,N} } )
--model  = MRTDHumieresD3Q15_2002("double_precision", { s1 = 1.6, s2 = 1.2, s4 = 1.6, s9 = 1/tau, s14 = 1.2, We = -1, Wej = 0, size = {N,N,N} } )
--model  = CLBM_D2Q9("double_precision", { w = 1/tau, wb = 1, w3 = 1.5, w4 = 1, size = {N,N} } )
--model  = CLBM_D3Q27("single_precision", { w = 1/tau, wb = 1, w3 = 1, w4 = 1, size = {N,N,N} } )

memory = MemoryManager()
kernel = KernelGenerator()
layout =
  ThreadLayout(
  {
    layout = "1D_BLOCK"
    --maxThreadsPerBlock = 512
  }
)
main = MainGenerator()

--main:setInitialisationMethod(DoubleShearLayer( { width = 80, amplitude = 0.05} ))
--main:setInitialisationMethod(TaylorGreenVortex( { A = 0.1, a = 1, B = 0.1, b = 1, C = 0.1, c = 1} ))
main:setInitialisationMethod(KidaVortex({U0 = 0.05}))

main:addModule(StatisticsGenerator({delay = 10.0, outputMode = "same_line"}))

main:addModule(
  OpenGLRenderGenerator(
    {
      FPS = 30,
      quantity = "velocity_norm",
      min = -0.02,
      max = 0.02,
      colors = "diverging",
      mode = "GPU",
      velocity_resolution = N / 32,
      velocity_scale = 0.2,
      velocity_mode = "arrow"
    }
  )
)

--main:addModule(WaitGenerator( { interval = 7400, duration = 10 } ))
--main:addModule(SavingGenerator({interval = 256000, quantities = { "density", "velocity" } }))
main:addModule(EarlyExitGenerator())

kernel:generate()
main:generate()

-- generate a diagram of the node velocities
--generateNodeDiagram(node)
--generateNodeDiagram(D3Q19Descriptor)

--TODO: function to compile the generated code
--os.execute("cd "..src_destination_folder)
--os.execute("./compile.sh")
--os.execute(src_destination_folder.."/compile.sh")
