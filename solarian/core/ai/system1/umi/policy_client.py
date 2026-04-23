"""
Policy Client Module

This module provides a client interface to communicate with the policy model server.
It handles serialization of observations and deserialization of actions.

Usage:
    from policy_client import PolicyClient
    
    client = PolicyClient(server_url="http://localhost:8001")
    action = client.predict(obs_dict)
"""

import requests
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PolicyClient:
    """Client for communicating with the policy model server"""
    
    def __init__(self, server_url: str = "http://localhost:8001", timeout: float = 30.0):
        """
        Initialize the policy client
        
        Args:
            server_url: URL of the policy server
            timeout: Request timeout in seconds
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._check_connection()
        
    def _check_connection(self):
        """Check if the server is reachable"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            logger.info(f"Successfully connected to policy server at {self.server_url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not connect to policy server: {e}")
            logger.warning("Make sure the server is running before making predictions")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        try:
            response = requests.get(f"{self.server_url}/model_info", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting model info: {e}")
            raise
    
    def get_shape_meta(self) -> Dict[str, Any]:
        """
        Get shape_meta configuration from the model
        
        Returns:
            Dictionary containing shape_meta configuration
        """
        try:
            response = requests.get(f"{self.server_url}/shape_meta", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting shape_meta: {e}")
            raise
    
    def reset(self):
        """Reset the policy state"""
        try:
            response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
            response.raise_for_status()
            logger.debug("Policy reset successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error resetting policy: {e}")
            raise
    
    def predict(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Predict action from observation
        
        Args:
            obs_dict: Dictionary of observations with numpy arrays
            
        Returns:
            Action as numpy array
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            obs_dict_serializable = {}
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    obs_dict_serializable[key] = value.tolist()
                else:
                    obs_dict_serializable[key] = value
            
            # Make prediction request
            payload = {"obs_dict": obs_dict_serializable}
            response = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            action = np.array(result['action'])
            
            return action
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise
    
    def predict_with_raw(self, obs_dict: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict action from observation, returning both processed and raw actions
        
        Args:
            obs_dict: Dictionary of observations with numpy arrays
            
        Returns:
            Tuple of (action, raw_action) as numpy arrays
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            obs_dict_serializable = {}
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    obs_dict_serializable[key] = value.tolist()
                else:
                    obs_dict_serializable[key] = value
            
            # Make prediction request
            payload = {"obs_dict": obs_dict_serializable}
            response = requests.post(
                f"{self.server_url}/predict",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            action = np.array(result['action'])
            raw_action = np.array(result['raw_action'])
            
            return action, raw_action
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during prediction: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if the server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5.0)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False


class PolicyClientWithFallback(PolicyClient):
    """
    Policy client with fallback to local model if server is unavailable
    
    This is useful for development and testing when you want to optionally
    use the server but fall back to local inference if it's not available.
    """
    
    def __init__(self, server_url: str = "http://localhost:8001", 
                 local_policy=None, local_device=None, timeout: float = 30.0):
        """
        Initialize the policy client with fallback
        
        Args:
            server_url: URL of the policy server
            local_policy: Local policy model to use as fallback
            local_device: Device for local inference
            timeout: Request timeout in seconds
        """
        self.local_policy = local_policy
        self.local_device = local_device
        self.use_server = True
        
        try:
            super().__init__(server_url, timeout)
        except Exception as e:
            logger.warning(f"Server not available, will use local model: {e}")
            self.use_server = False
            if local_policy is None:
                raise ValueError("Server unavailable and no local policy provided")
    
    def predict(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict with fallback to local model"""
        if self.use_server and self.health_check():
            try:
                return super().predict(obs_dict)
            except Exception as e:
                logger.warning(f"Server prediction failed, falling back to local: {e}")
                self.use_server = False
        
        # Use local model
        if self.local_policy is None:
            raise RuntimeError("Server unavailable and no local policy provided")
        
        from diffusion_policy.common.pytorch_util import dict_apply
        import torch
        
        with torch.no_grad():
            obs_dict_torch = dict_apply(obs_dict, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.local_device))
            result = self.local_policy.predict_action(obs_dict_torch)
            action = result['action_pred'][0].detach().to('cpu').numpy()
        
        return action
