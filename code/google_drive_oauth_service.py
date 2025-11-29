import os
import io
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

class GoogleDriveOAuthService:
    def __init__(self, credentials_file='credentials.json', token_file='token.json', scopes=None):
        """
        Initialize Google Drive service with OAuth 2.0 credentials

        Args:
            credentials_file (str): Path to OAuth 2.0 client secrets JSON file
            token_file (str): Path to token file for storing access tokens
            scopes (list): List of scopes for authentication
        """
        if scopes is None:
            scopes = ['https://www.googleapis.com/auth/drive']

        self.credentials_file = credentials_file
        self.token_file = token_file
        self.scopes = scopes
        self.creds = None

        try:
            # Load or refresh credentials
            self.creds = self._get_credentials()
            
            # Build the Drive API service
            self.service = build('drive', 'v3', credentials=self.creds)
            print("[SUCCESS] Google Drive OAuth service initialized successfully")

        except Exception as e:
            print(f"[ERROR] Failed to initialize Google Drive OAuth service: {e}")
            raise

    def _get_credentials(self):
        """
        Get valid credentials, refreshing if necessary
        """
        creds = None
        
        # Check if token file exists
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds

    def list_files(self, folder_id=None, query=None, page_size=100):
        """
        List files in Google Drive

        Args:
            folder_id (str): ID of the folder to list files from
            query (str): Custom query string
            page_size (int): Number of files to return per page

        Returns:
            list: List of file metadata
        """
        try:
            if query is None:
                if folder_id:
                    query = f"'{folder_id}' in parents and trashed=false"
                else:
                    query = "trashed=false"

            results = self.service.files().list(
                q=query,
                pageSize=page_size,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)"
            ).execute()

            files = results.get('files', [])
            print(f"[INFO] Found {len(files)} files")
            return files

        except HttpError as e:
            print(f"[ERROR] Error listing files: {e}")
            return []

    def upload_file(self, file_path, file_name=None, folder_id=None, mime_type=None):
        """
        Upload a file to Google Drive

        Args:
            file_path (str): Local path to the file
            file_name (str): Name for the file in Drive (optional)
            folder_id (str): ID of the folder to upload to (optional)
            mime_type (str): MIME type of the file (optional)

        Returns:
            dict: File metadata if successful, None if failed
        """
        try:
            if not os.path.exists(file_path):
                print(f"[ERROR] File not found: {file_path}")
                return None

            # Get file name if not provided
            if file_name is None:
                file_name = os.path.basename(file_path)

            # Determine MIME type
            if mime_type is None:
                import mimetypes
                mime_type, _ = mimetypes.guess_type(file_path)
                if mime_type is None:
                    mime_type = 'application/octet-stream'

            # Create file metadata
            file_metadata = {
                'name': file_name
            }

            # Add parent folder if specified
            if folder_id:
                file_metadata['parents'] = [folder_id]

            # Create media upload object
            media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

            # Upload file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,mimeType,size,modifiedTime,webViewLink'
            ).execute()

            print(f"[SUCCESS] File uploaded successfully: {file_name} (ID: {file.get('id')})")
            return file

        except HttpError as e:
            print(f"[ERROR] Error uploading file: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error uploading file: {e}")
            return None

    def download_file(self, file_id, local_path=None, file_name=None):
        """
        Download a file from Google Drive

        Args:
            file_id (str): ID of the file to download
            local_path (str): Local directory to save the file (optional)
            file_name (str): Name for the downloaded file (optional)

        Returns:
            str: Path to downloaded file if successful, None if failed
        """
        try:
            # Get file metadata
            file_metadata = self.service.files().get(fileId=file_id).execute()
            file_name = file_name or file_metadata.get('name', 'downloaded_file')

            # Set download path
            if local_path:
                os.makedirs(local_path, exist_ok=True)
                download_path = os.path.join(local_path, file_name)
            else:
                download_path = file_name

            # Download file
            request = self.service.files().get_media(fileId=file_id)
            with io.FileIO(download_path, 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"[DOWNLOAD] Download {int(status.progress() * 100)}%.")

            print(f"[SUCCESS] File downloaded successfully: {download_path}")
            return download_path

        except HttpError as e:
            print(f"[ERROR] Error downloading file: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error downloading file: {e}")
            return None

    def create_folder(self, folder_name, parent_id=None):
        """
        Create a new folder in Google Drive

        Args:
            folder_name (str): Name of the folder
            parent_id (str): ID of the parent folder (optional)

        Returns:
            dict: Folder metadata if successful, None if failed
        """
        try:
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }

            if parent_id:
                file_metadata['parents'] = [parent_id]

            folder = self.service.files().create(
                body=file_metadata,
                fields='id,name,mimeType,modifiedTime'
            ).execute()

            print(f"[INFO] Folder created successfully: {folder_name} (ID: {folder.get('id')})")
            return folder

        except HttpError as e:
            print(f"[ERROR] Error creating folder: {e}")
            return None

    def delete_file(self, file_id):
        """
        Delete a file from Google Drive

        Args:
            file_id (str): ID of the file to delete

        Returns:
            bool: True if successful, False if failed
        """
        try:
            self.service.files().delete(fileId=file_id).execute()
            print(f"[DELETE] File deleted successfully: {file_id}")
            return True

        except HttpError as e:
            print(f"[ERROR] Error deleting file: {e}")
            return False

    def search_files(self, query, page_size=100):
        """
        Search for files in Google Drive

        Args:
            query (str): Search query
            page_size (int): Number of results to return

        Returns:
            list: List of matching files
        """
        try:
            results = self.service.files().list(
                q=query,
                pageSize=page_size,
                fields="files(id, name, mimeType, size, modifiedTime, parents)"
            ).execute()

            files = results.get('files', [])
            print(f"[SEARCH] Search found {len(files)} files")
            return files

        except HttpError as e:
            print(f"[ERROR] Error searching files: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize service
    drive_service = GoogleDriveOAuthService()

    # Example: List files in root
    files = drive_service.list_files()
    for file in files[:5]:  # Show first 5 files
        print(f"- {file['name']} ({file['id']})")

    # Example: Upload a file
    # uploaded_file = drive_service.upload_file('path/to/your/file.txt')
    # if uploaded_file:
    #     print(f"Uploaded: {uploaded_file['name']}")

    # Example: Download a file
    # downloaded_path = drive_service.download_file('file_id_here')
    # if downloaded_path:
    #     print(f"Downloaded to: {downloaded_path}")