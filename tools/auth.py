
import boto3
import gradio as gr
import hmac
import hashlib
import base64
from tools.helper_functions import get_or_create_env_var

client_id = get_or_create_env_var('AWS_CLIENT_ID', '')
#print(f'The value of AWS_CLIENT_ID is {client_id}')

client_secret = get_or_create_env_var('AWS_CLIENT_SECRET', '')
#print(f'The value of AWS_CLIENT_SECRET is {client_secret}')

user_pool_id = get_or_create_env_var('AWS_USER_POOL_ID', '')
#print(f'The value of AWS_USER_POOL_ID is {user_pool_id}')

def calculate_secret_hash(client_id, client_secret, username):
    message = username + client_id
    dig = hmac.new(
        str(client_secret).encode('utf-8'),
        msg=str(message).encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    secret_hash = base64.b64encode(dig).decode()
    return secret_hash

def authenticate_user(username:str, password:str, user_pool_id:str=user_pool_id, client_id:str=client_id, client_secret:str=client_secret):
    """Authenticates a user against an AWS Cognito user pool.

    Args:
        user_pool_id (str): The ID of the Cognito user pool.
        client_id (str): The ID of the Cognito user pool client.
        username (str): The username of the user.
        password (str): The password of the user.
        client_secret (str): The client secret of the app client

    Returns:
        bool: True if the user is authenticated, False otherwise.
    """

    client = boto3.client('cognito-idp')  # Cognito Identity Provider client

    # Compute the secret hash
    secret_hash = calculate_secret_hash(client_id, client_secret, username)

    try:

        if client_secret == '':
            response = client.initiate_auth(
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password,
                },
                ClientId=client_id
            )

        else:
            response = client.initiate_auth(
            AuthFlow='USER_PASSWORD_AUTH',
            AuthParameters={
                'USERNAME': username,
                'PASSWORD': password,
                'SECRET_HASH': secret_hash
            },
            ClientId=client_id
            )

        # If successful, you'll receive an AuthenticationResult in the response
        if response.get('AuthenticationResult'):
            return True
        else:
            return False

    except client.exceptions.NotAuthorizedException:
        return False
    except client.exceptions.UserNotFoundException:
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False 
