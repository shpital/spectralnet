import torch
import logging
from src.spectralnet.core.layers.spectral_remizov_layer import SpectralRemizovLayer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Gradcheck")

def main():
    logger.info("Starting isolated gradcheck for SpectralRemizovLayer...")
    
    channels = 4
    modes = 4
    spatial_res = 8
    
    # Layer MUST be in double (float64)
    layer = SpectralRemizovLayer(channels=channels, modes=modes).double()
    
    # Input tensor: float64 and requires_grad=True
    x = torch.randn(
        2, channels, spatial_res, spatial_res, 
        dtype=torch.float64, 
        requires_grad=True
    )

    try:
        test_passed = torch.autograd.gradcheck(
            layer, 
            (x,), 
            eps=1e-6, 
            atol=1e-4,
            nondet_tol=1e-7  
        )
        
        if test_passed:
            logger.info("✅ SUCCESS: SpectralRemizovLayer passed gradcheck.")
            logger.info("The math of complex FFT and W(k) masking is correctly differentiable.")
    except Exception as e:
        logger.error(f"❌ FAILURE: gradcheck found an error in gradients.\n{e}")

if __name__ == "__main__":
    main()