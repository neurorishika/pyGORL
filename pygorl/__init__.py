
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        'cogpolicy',
        'cogq',
        'rdp_client',
    },
    submod_attrs={
        'cogpolicy': [
            'ACLPolicyGradient',
            'AdvLPolicyGradient',
            'VLPolicyGradient',
            'VSPolicyGradient',
        ],
        'cogq': [
            'FQLearning',
            'HetFQLearning',
            'HetOSFQLearning',
            'HetOSQLearning',
            'HetQLearning',
            'HetSOSFQLearning',
            'OSFQLearning',
            'OSQLearning',
            'QLearning',
            'SOSFQLearning',
        ],
        'rdp_client': [
            'unlock_and_unzip_file',
            'zip_and_lock_folder',
        ],
    },
)

__all__ = ['ACLPolicyGradient', 'AdvLPolicyGradient', 'FQLearning',
           'HetFQLearning', 'HetOSFQLearning', 'HetOSQLearning',
           'HetQLearning', 'HetSOSFQLearning', 'OSFQLearning', 'OSQLearning',
           'QLearning', 'SOSFQLearning', 'VLPolicyGradient',
           'VSPolicyGradient', 'cogpolicy', 'cogq', 'rdp_client',
           'unlock_and_unzip_file', 'zip_and_lock_folder']
