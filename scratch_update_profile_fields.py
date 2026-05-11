import re

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Profile States
old_profile_states = """  // Profile states
  const [profileUsername, setProfileUsername] = useState('');
  const [profilePassword, setProfilePassword] = useState('');"""

new_profile_states = """  // Profile states
  const [profileUsername, setProfileUsername] = useState('');
  const [profilePassword, setProfilePassword] = useState('');
  const [profileEmail, setProfileEmail] = useState('');
  const [profilePhone, setProfilePhone] = useState('');"""

code = code.replace(old_profile_states, new_profile_states)


# 2. loadData
old_load = """      if (me) {
        setProfileUsername(me.username);
        setProfilePassword(me.password || '');
      }"""

new_load = """      if (me) {
        setProfileUsername(me.username);
        setProfilePassword(me.password || '');
        setProfileEmail(me.email || '');
        setProfilePhone(me.phone || '');
      }"""

code = code.replace(old_load, new_load)


# 3. handleSubmit Validation
old_submit = """    if (!username || !password) {
      addNotification({ type: 'error', message: 'Please provide both username and password' });"""

new_submit = """    if (!username || !password || !email || !phone) {
      addNotification({ type: 'error', message: 'Please provide all 4 fields (username, password, email, phone)' });"""

code = code.replace(old_submit, new_submit)


# 4. handleProfileUpdate Validation
old_profile_update = """    if (!profileUsername || !profilePassword) {
      addNotification({ type: 'error', message: 'Please provide both username and password' });"""

new_profile_update = """    if (!profileUsername || !profilePassword || !profileEmail || !profilePhone) {
      addNotification({ type: 'error', message: 'Please provide all 4 fields (username, password, email, phone)' });"""

code = code.replace(old_profile_update, new_profile_update)


# 5. handleProfileUpdate Save
old_profile_save = "storedData[meIndex] = { ...storedData[meIndex], username: profileUsername, password: profilePassword };"
new_profile_save = "storedData[meIndex] = { ...storedData[meIndex], username: profileUsername, password: profilePassword, email: profileEmail, phone: profilePhone };"

code = code.replace(old_profile_save, new_profile_save)


# 6. Profile JSX
old_profile_jsx = """                    <div>
                      <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Password</label>
                      <div className="relative">
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={profilePassword}
                          onChange={(e) => setProfilePassword(e.target.value)}
                          className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors pr-10"
                          placeholder="Enter password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                        >
                          {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                    <div className="pt-4">"""

new_profile_jsx = """                    <div>
                      <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Email</label>
                      <input
                        type="email"
                        value={profileEmail}
                        onChange={(e) => setProfileEmail(e.target.value)}
                        className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                        placeholder="Enter email id"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Phone Number</label>
                      <input
                        type="text"
                        value={profilePhone}
                        onChange={(e) => setProfilePhone(e.target.value.replace(/\D/g, '').slice(0, 10))}
                        pattern="[0-9]{10}"
                        maxLength={10}
                        className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors"
                        placeholder="Enter 10-digit phone number"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-[var(--text-secondary)] mb-1">Password</label>
                      <div className="relative">
                        <input
                          type={showPassword ? 'text' : 'password'}
                          value={profilePassword}
                          onChange={(e) => setProfilePassword(e.target.value)}
                          className="w-full bg-[var(--bg)] border border-[var(--border)] rounded-lg px-4 py-2 text-[var(--text)] focus:outline-none focus:border-[var(--primary)] transition-colors pr-10"
                          placeholder="Enter password"
                        />
                        <button
                          type="button"
                          onClick={() => setShowPassword(!showPassword)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text)] transition-colors"
                        >
                          {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                        </button>
                      </div>
                    </div>
                    <div className="pt-4">"""

code = code.replace(old_profile_jsx, new_profile_jsx)

with open('e:/aitrade/frontend/src/pages/AdminDashboard.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
