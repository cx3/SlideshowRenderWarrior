import hashlib
import weakref
import datetime
from typing import List, Dict, Any


class UuidController:
    __attached: List[Dict] = []

    @staticmethod
    def set_uid(some: Any) -> weakref.proxy:

        def _next_uid():
            sha256 = hashlib.sha256()
            sha256.update(bytearray(datetime.datetime.now().strftime("%Y-%m-%d @ %H:%M:%S")))
            return str(sha256.digest())[2:-1]  # trim prefix b' and postfix '

        if not hasattr(some, 'attached_uid'):
            proxy = weakref.proxy(some)

            for index, next_dict in enumerate(UuidController.__attached):
                if proxy == next_dict['proxy']:
                    setattr(some, 'attached_uid', next_dict['uid'])
                    return next_dict['proxy']
            uid = _next_uid()
            UuidController.__attached.append({
                'attached_uid': uid,
                'proxy': proxy
            })
            return UuidController.__attached[-1]['proxy']
        else:
            attached_uid = getattr(some, 'attached_uid')
            for index, next_dict in enumerate(UuidController.__attached):
                if attached_uid == next_dict['attached_uid']:
                    return next_dict['proxy']

    @staticmethod
    def get_by_uid(uid: str) -> weakref.proxy or None:
        for _ in UuidController.__attached:
            if uid == getattr(_, 'attached_uid'):
                return _['proxy']
        return None
